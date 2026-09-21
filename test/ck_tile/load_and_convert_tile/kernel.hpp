// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

template <typename BlockWarps, typename BlockTile, typename WarpTile, typename Vector>
struct LoadAndConvertShape
{
    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t Warp_M = WarpTile::at(number<0>{});
    static constexpr index_t Warp_N = WarpTile::at(number<1>{});

    static constexpr index_t Vector_N = Vector::at(number<1>{});

    static constexpr index_t WarpPerBlock_M = BlockWarps::at(number<0>{});
    static constexpr index_t WarpPerBlock_N = BlockWarps::at(number<1>{});

    static constexpr index_t Repeat_M = Block_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N / (WarpPerBlock_N * Warp_N);

    static constexpr index_t BlockSize =
        ck_tile::get_warp_size() * reduce_on_sequence(BlockWarps{}, multiplies<>{}, number<1>{});
};

template <typename XDataType_, typename YDataType_, typename BlockShape_, typename LoadTranspose_>
struct LoadAndConvertProblem
{
    using XDataType     = remove_cvref_t<XDataType_>;
    using YDataType     = remove_cvref_t<YDataType_>;
    using BlockShape    = remove_cvref_t<BlockShape_>;
    using LoadTranspose = remove_cvref_t<LoadTranspose_>;
};

template <typename Problem_>
struct LoadAndConvertPolicy
{
    using Problem       = ck_tile::remove_cvref_t<Problem_>;
    using XDataType     = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using YDataType     = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using LoadTranspose = ck_tile::remove_cvref_t<typename Problem::LoadTranspose>;

    template <index_t NumAccess, typename Problem>
    CK_TILE_DEVICE static constexpr auto get_warp_dstr_encoding()
    {
        using S = typename Problem::BlockShape;

        if constexpr(NumAccess == 1)
            return tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<get_warp_size() * S::Vector_N / S::Block_N>,
                      sequence<S::Block_N / S::Vector_N, S::Vector_N>>,
                tuple<sequence<2, 1>>,
                tuple<sequence<0, 0>>,
                sequence<2>,
                sequence<1>>{};
        else
            return tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<get_warp_size() * S::Vector_N / S::Block_N>,
                      sequence<S::Block_N / S::Vector_N, NumAccess, S::Vector_N / NumAccess>>,
                tuple<sequence<2, 1>>,
                tuple<sequence<0, 0>>,
                sequence<2, 2>,
                sequence<1, 2>>{};
    }

    template <typename DataType>
    CK_TILE_DEVICE static constexpr auto GetVectorSize()
    {
        return DS_READ_TR_SIZE() / sizeof(DataType);
    }

    template <typename Problem, typename DataType>
    CK_TILE_DEVICE static constexpr auto MakeDRAMDistribution()
    {
        using S = typename Problem::BlockShape;

        constexpr index_t thread_elements = S::Warp_M * S::Warp_N / get_warp_size();
        constexpr index_t NumAccess =
            LoadTranspose::value ? thread_elements / GetVectorSize<DataType>() : 1;

        constexpr auto a_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<S::Block_M / S::WarpPerBlock_M * S::Block_N / S::WarpPerBlock_N /
                               get_warp_size() / S::Vector_N,
                           S::WarpPerBlock_M * S::WarpPerBlock_N>,
                  sequence<>>,
            tuple<sequence<1>>,
            tuple<sequence<1>>,
            sequence<1>,
            sequence<0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encode, get_warp_dstr_encoding<NumAccess, Problem>());

        return make_static_tile_distribution(a_block_dstr_encode);
    }

    template <typename Problem, typename DataType>
    CK_TILE_DEVICE static constexpr auto MakeDRAMTransposedDistribution()
    {
        return make_static_tile_distribution(
            typename InputTileDistributionTraits<
                typename decltype(MakeDRAMDistribution<Problem, DataType>())::DstrEncode,
                DataType>::TransposedDstrEncode{});
    }
};

template <typename Problem_, typename Policy_>
struct LoadAndConvertKernel
{
    using Problem       = ck_tile::remove_cvref_t<Problem_>;
    using XDataType     = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using YDataType     = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using Policy        = ck_tile::remove_cvref_t<Policy_>;
    using LoadTranspose = ck_tile::remove_cvref_t<typename Problem::LoadTranspose>;

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    CK_TILE_HOST static auto BlockSize() { return kBlockSize; }

    private:
    CK_TILE_DEVICE static constexpr auto get_block_dims()
    {
        using S = typename Problem::BlockShape;
        return make_tuple(S::Block_M, S::Block_N);
    }

    template <bool kTranspose>
    CK_TILE_DEVICE static constexpr auto get_lds_block_strides()
    {
        using S = typename Problem::BlockShape;
        if constexpr(kTranspose)
        {
            return make_tuple(1, S::Block_M);
        }
        else
        {
            return make_tuple(S::Block_N, 1);
        }
    }

    template <bool kTranspose>
    CK_TILE_DEVICE static auto make_global_strides(index_t M, index_t N)
    {
        if constexpr(kTranspose)
        {
            return make_tuple(index_t{1}, M);
        }
        else
        {
            return make_tuple(N, index_t{1});
        }
    }

    template <bool kTranspose>
    CK_TILE_DEVICE static auto make_lds_naive_view(XDataType* a_lds)
    {
        constexpr auto block_dims    = get_block_dims();
        constexpr auto block_strides = get_lds_block_strides<kTranspose>();
        using Shape                  = typename Problem::BlockShape;
        if constexpr(kTranspose)
        {
            return make_naive_tensor_view<address_space_enum::lds>(
                a_lds, block_dims, block_strides, number<1>{}, number<1>{});
        }
        else
        {
            return make_naive_tensor_view<address_space_enum::lds>(
                a_lds, block_dims, block_strides, number<Shape::Vector_N>{}, number<1>{});
        }
    }

    CK_TILE_DEVICE static auto make_lds_write_window(const auto& a_lds_naive_view)
    {
        constexpr auto block_dims = get_block_dims();
        return make_tile_window(a_lds_naive_view, block_dims, {0, 0});
    }

    template <bool kTranspose>
    CK_TILE_DEVICE static auto make_lds_read_window(XDataType* a_lds, const auto& a_lds_naive_view)
    {
        using S                   = typename Problem::BlockShape;
        constexpr auto block_dims = get_block_dims();
        if constexpr(kTranspose)
        {
            constexpr auto block_dims_t    = make_tuple(S::Block_N, S::Block_M);
            constexpr auto block_strides_t = make_tuple(S::Block_M, 1);
            auto a_lds_transpose_view      = make_naive_tensor_view<address_space_enum::lds>(
                a_lds,
                block_dims_t,
                block_strides_t,
                number<Policy::template GetVectorSize<XDataType>()>{},
                number<1>{});
            return make_tile_window(
                a_lds_transpose_view,
                block_dims_t,
                {0, 0},
                Policy::template MakeDRAMTransposedDistribution<Problem, XDataType>());
        }
        else
        {
            return make_tile_window(a_lds_naive_view,
                                    block_dims,
                                    {0, 0},
                                    Policy::template MakeDRAMDistribution<Problem, XDataType>());
        }
    }

    template <bool kTranspose>
    CK_TILE_DEVICE static auto make_a_dram_block_window(
        const XDataType* a, index_t M, index_t N, index_t m_block_base, const auto& global_strides)
    {
        constexpr auto block_dims = get_block_dims();
        if constexpr(kTranspose)
        {
            const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
                a, make_tuple(M, N), global_strides, number<1>{}, number<1>{});
            return make_tile_window(a_tensor,
                                    block_dims,
                                    {m_block_base, 0},
                                    Policy::template MakeDRAMDistribution<Problem, XDataType>());
        }
        else
        {
            using Shape         = typename Problem::BlockShape;
            const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
                a, make_tuple(M, N), global_strides, number<Shape::Vector_N>{}, number<1>{});
            return make_tile_window(a_tensor,
                                    block_dims,
                                    {m_block_base, 0},
                                    Policy::template MakeDRAMDistribution<Problem, XDataType>());
        }
    }

    template <bool kTranspose>
    CK_TILE_DEVICE static auto make_c_dram_block_window(
        YDataType* c, index_t M, index_t N, index_t m_block_base, const auto& global_strides)
    {
        constexpr auto block_dims = get_block_dims();
        if constexpr(kTranspose)
        {
            const auto c_tensor = make_naive_tensor_view<address_space_enum::global>(
                c, make_tuple(M, N), global_strides, number<1>{}, number<1>{});
            return make_tile_window(c_tensor,
                                    block_dims,
                                    {m_block_base, 0},
                                    Policy::template MakeDRAMDistribution<Problem, YDataType>());
        }
        else
        {
            using Shape         = typename Problem::BlockShape;
            const auto c_tensor = make_naive_tensor_view<address_space_enum::global>(
                c, make_tuple(M, N), global_strides, number<Shape::Vector_N>{}, number<1>{});
            return make_tile_window(c_tensor,
                                    block_dims,
                                    {m_block_base, 0},
                                    Policy::template MakeDRAMDistribution<Problem, YDataType>());
        }
    }

    public:
    CK_TILE_DEVICE void operator()(const XDataType* a, YDataType* c, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        constexpr bool kTransposePath = LoadTranspose::value;

        // LDS buffer
        __shared__ XDataType a_lds[S::Block_M * S::Block_N];

        const index_t m_block_base = __builtin_amdgcn_readfirstlane(get_block_id() * S::Block_M);
        const auto global_strides  = make_global_strides<kTransposePath>(M, N);

        auto a_lds_view               = make_lds_naive_view<kTransposePath>(a_lds);
        auto a_block_lds_write_window = make_lds_write_window(a_lds_view);
        auto a_block_lds_read_window  = make_lds_read_window<kTransposePath>(a_lds, a_lds_view);
        auto a_block_window =
            make_a_dram_block_window<kTransposePath>(a, M, N, m_block_base, global_strides);
        auto c_block_window =
            make_c_dram_block_window<kTransposePath>(c, M, N, m_block_base, global_strides);

        const index_t num_n_loops = integer_divide_ceil(N, S::Block_N);
        for(index_t n_iter = 0; n_iter < num_n_loops; ++n_iter)
        {
            auto dram_tile = load_tile(a_block_window);
            store_tile(a_block_lds_write_window, dram_tile);
            block_sync_lds();

            decltype(load_tile(c_block_window)) c_tile;
            load_and_convert_tile<8, LoadTranspose::value>(c_tile, a_block_lds_read_window);
            store_tile(c_block_window, c_tile);
            block_sync_lds();

            if(n_iter < num_n_loops - 1)
            {
                move_tile_window(a_block_window, {0, S::Block_N});
                move_tile_window(c_block_window, {0, S::Block_N});
            }
        }
    }
};

} // namespace ck_tile
