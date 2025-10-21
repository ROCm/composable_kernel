#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <index_t M_Tile_, index_t N_Tile_, index_t M_Warp_, index_t N_Warp_>
struct TileShape
{
    static constexpr index_t M        = M_Tile_;
    static constexpr index_t N        = N_Tile_;
    static constexpr index_t Mw       = M_Warp_;
    static constexpr index_t Nw       = N_Warp_;
    static constexpr index_t NumWarps = Mw * Nw;
};

template <typename DataType_, typename TileShape_>
struct AsyncLSPolicy
{
    using DataType = remove_cvref_t<DataType_>;
    using Shape    = remove_cvref_t<TileShape_>;
    using RawType =
        std::conditional_t<std::is_class_v<DataType>, typename DataType::type, DataType>;

    constexpr static index_t max_vector_size = 16;
    constexpr static index_t warp_size       = 64;
    constexpr static index_t PackedSize      = numeric_traits<DataType>::PackedSize;
    constexpr static index_t kMPerBlock      = Shape::M;
    constexpr static index_t kNPerBlock      = Shape::N / PackedSize;

    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {

        constexpr auto lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                         make_tuple(number<kNPerBlock>{}, number<1>{}),
                                         number<16>{},
                                         number<1>{});

        return lds_block_desc_0;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeDRAMDistribution()
    {
        static_assert(Shape::Nw == 1);
        constexpr index_t N1 = max_vector_size / sizeof(RawType);
        constexpr index_t N0 = kNPerBlock / N1;
        constexpr index_t M2 = warp_size / N0;
        constexpr index_t M1 = Shape::Mw;
        constexpr index_t M0 = Shape::M / (M1 * M2);

        constexpr auto encoding =
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<N0, N1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{};
        return make_static_tile_distribution(encoding);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return sizeof(RawType) * MakeLdsBlockDescriptor().get_element_space_size();
    }
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSize()
    {
        return std::min(static_cast<int>(kNPerBlock),
                        static_cast<int>(max_vector_size / sizeof(RawType)));
    }
};

struct AsyncLSKernelArgs
{
    void* a_ptr;
    void* b_ptr;
    index_t M;
    index_t N;
    index_t stride_A;
    index_t stride_B;
};

template <typename Policy_>
struct AsyncLSKernel
{
    using Policy   = remove_cvref_t<Policy_>;
    using Shape    = typename Policy::Shape;
    using DataType = typename Policy::DataType;
    using RawType  = typename Policy::RawType;

    constexpr static int kBlockPerCu    = 1;
    constexpr static index_t kBlockSize = Shape::NumWarps * get_warp_size();
    constexpr static index_t PackedSize = Policy::PackedSize;

    CK_TILE_HOST static dim3 BlockSize() { return dim3(kBlockSize); }
    CK_TILE_HOST static dim3 GridSize(index_t M, index_t N)
    {
        const index_t GridDimX = (M + Shape::M - 1) / Shape::M;
        const index_t GridDimY = (N + Shape::N - 1) / Shape::N;
        return dim3(GridDimX, GridDimY, 1);
    }

    CK_TILE_DEVICE auto operator()(AsyncLSKernelArgs kargs) const -> void
    {
        const index_t i_m = amd_wave_read_first_lane(blockIdx.x * Policy::kMPerBlock);
        const index_t i_n = amd_wave_read_first_lane(blockIdx.y * Policy::kNPerBlock);

        RawType* a_ptr = static_cast<RawType*>(kargs.a_ptr);
        RawType* b_ptr = static_cast<RawType*>(kargs.b_ptr);

        // allocate LDS
        __shared__ RawType smem_ptr_0[Policy::GetSmemSize()];

        const auto& a_tensor_view = make_naive_tensor_view<address_space_enum::global>(
            a_ptr,
            make_tuple(kargs.M, kargs.N / PackedSize),
            make_tuple(kargs.stride_A / PackedSize, 1),
            number<Policy::GetVectorSize()>{},
            number<1>{});

        const auto& b_tensor_view =
            make_naive_tensor_view<address_space_enum::global, memory_operation_enum::set>(
                b_ptr,
                make_tuple(kargs.M, kargs.N / PackedSize),
                make_tuple(kargs.stride_B / PackedSize, 1),
                number<Policy::GetVectorSize()>{},
                number<1>{});

        auto a_block_window =
            make_tile_window(a_tensor_view,
                             make_tuple(number<Policy::kMPerBlock>{}, number<Policy::kNPerBlock>{}),
                             {i_m, i_n},
                             Policy::MakeDRAMDistribution());

        auto b_block_window =
            make_tile_window(b_tensor_view,
                             make_tuple(number<Policy::kMPerBlock>{}, number<Policy::kNPerBlock>{}),
                             {i_m, i_n});

        auto lds_0_tensor_view =
            make_tensor_view<address_space_enum::lds>(smem_ptr_0, Policy::MakeLdsBlockDescriptor());
        auto lds_0_window =
            make_tile_window(lds_0_tensor_view,
                             make_tuple(number<Policy::kMPerBlock>{}, number<Policy::kNPerBlock>{}),
                             {0, 0},
                             Policy::MakeDRAMDistribution());
#if 0
        auto dram_tile = load_tile(a_block_window);
        store_tile(lds_0_window, dram_tile);
#else
        async_load_tile(lds_0_window, a_block_window);
#endif
        block_sync_lds();
        auto lds_tile = load_tile(lds_0_window);
        store_tile(b_block_window, lds_tile);
    }
};
} // namespace ck_tile
