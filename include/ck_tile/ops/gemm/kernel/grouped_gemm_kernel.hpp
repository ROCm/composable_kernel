// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/core/utility/amd_address_space.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host.hpp"

namespace ck_tile {

struct GroupedGemmHostArgs
{
    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GroupedGemmKernel
{
    using TilePartitioner                    = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline                       = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline                   = remove_cvref_t<EpiloguePipeline_>;
    using ALayout                            = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout                            = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout                            = remove_cvref_t<typename GemmPipeline::CLayout>;
    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    struct GemmTransKernelArg
    {
        GroupedGemmHostArgs group_karg;
        ck_tile::index_t block_start;
        ck_tile::index_t block_end;

        GemmTransKernelArg() = default;
        GemmTransKernelArg(GroupedGemmHostArgs&& karg, index_t bl_start, index_t bl_end)
            : group_karg{karg}, block_start{bl_start}, block_end{bl_end}
        {
        }
    };

    __host__ static size_t GetWorkSpaceSize(const std::vector<GroupedGemmHostArgs>& gemm_descs)
    {
        return gemm_descs.size() * sizeof(GemmTransKernelArg);
    }

    __host__ static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    using Hargs = GroupedGemmHostArgs;

    __host__ static constexpr auto GridSize(const std::vector<Hargs>& gemm_descs)
    {
        index_t grid_size = 0;
        for(const auto& it_desc : gemm_descs)
        {
            const auto dim3 = TilePartitioner::GridSize(it_desc.M, it_desc.N);
            grid_size += dim3.x * dim3.y * 1;
        }
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto MakeKargs(const std::vector<Hargs>& gemm_descs)
    {
        std::vector<GemmTransKernelArg> gemm_kernel_args_;
        index_t group_count = ck_tile::type_convert<ck_tile::index_t>(gemm_descs.size());
        index_t grid_size   = 0;
        gemm_kernel_args_.reserve(group_count);

        for(std::size_t i = 0; i < gemm_descs.size(); ++i)
        {
            const index_t M = gemm_descs[i].M;
            const index_t N = gemm_descs[i].N;
            const index_t K = gemm_descs[i].K;

            if(M == 0 || N == 0 || K == 0)
            {
                continue;
            }

            const index_t stride_a = gemm_descs[i].stride_A;
            const index_t stride_b = gemm_descs[i].stride_B;
            const index_t stride_c = gemm_descs[i].stride_C;

            const auto dim3             = TilePartitioner::GridSize(M, N);
            const index_t grid_size_grp = dim3.x * 1 * 1;

            const index_t block_start = grid_size;
            const index_t block_end   = grid_size + grid_size_grp;

            grid_size += grid_size_grp;

            auto karg = GroupedGemmHostArgs{type_convert<const ADataType*>(gemm_descs[i].a_ptr),
                                            type_convert<const BDataType*>(gemm_descs[i].b_ptr),
                                            type_convert<CDataType*>(gemm_descs[i].c_ptr),
                                            M,
                                            N,
                                            K,
                                            stride_a,
                                            stride_b,
                                            stride_c};

            gemm_kernel_args_.emplace_back(std::move(karg), block_start, block_end);
        }

        return gemm_kernel_args_;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void Run(const Hargs& kargs, const index_t block_start) const
    {
        const auto [i_m, i_n] = TilePartitioner{}(block_start, kargs.N);
        // options
        const ADataType* a_start = static_cast<const ADataType*>(kargs.a_ptr);
        const BDataType* b_start = static_cast<const BDataType*>(kargs.b_ptr);
        // Convert pointers to tensor views
        auto a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::VectorSizeA>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(1, kargs.stride_A),
                    number<1>{},
                    number<1>{});
            }
        }();

        auto b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
                    make_tuple(kargs.N, kargs.K),
                    make_tuple(1, kargs.stride_B),
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
                    make_tuple(kargs.N, kargs.K),
                    make_tuple(kargs.stride_B, 1),
                    number<GemmPipeline::VectorSizeB>{},
                    number<1>{});
            }
        }();

        auto a_pad_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, false>{});
            }
        }();
        // clang-format on

        auto a_block_window = make_tile_window(
            a_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            {i_m, 0});

        auto b_pad_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::NPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::NPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<GemmPipeline::kPadN, false>{});
            }
        }();

        auto b_block_window = make_tile_window(
            b_pad_view,
            make_tuple(number<TilePartitioner::NPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            {i_n, 0});

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        const index_t num_loop = TilePartitioner::GetLoopNum(kargs.K);

        // Run GEMM cooperatively by whole wokrgroup.
        auto c_block_tile =
            GemmPipeline{}.template operator()(a_block_window, b_block_window, num_loop, smem_ptr);

        CDataType* c_start = static_cast<CDataType*>(kargs.c_ptr);
        auto c_tensor_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    c_start,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<GemmPipeline::VectorSizeC>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    c_start,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<1>{},
                    number<1>{});
            }
        }();

        auto c_pad_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, false>{});
            }
        }();
        auto CBlockWindow_pad = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        EpiloguePipeline{}(CBlockWindow_pad, c_block_tile);
    }

    CK_TILE_DEVICE void operator()(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   int group_count) const
    {
        const index_t block_id   = ck_tile::get_block_1d_id();
        const auto gemm_desc_ptr = reinterpret_cast<const GemmTransKernelArg*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        index_t left     = 0;
        index_t right    = group_count;
        index_t group_id = index_t((left + right) / 2);

        while((!(block_id >= gemm_desc_ptr[group_id].block_start &&
                 block_id < gemm_desc_ptr[group_id].block_end)) &&
              left <= right)
        {
            if(block_id < gemm_desc_ptr[group_id].block_start)
            {
                right = group_id;
            }
            else
            {
                left = group_id;
            }
            group_id = index_t((left + right) / 2);
        }

        Run(gemm_desc_ptr[group_id].group_karg, gemm_desc_ptr[group_id].block_start);
    }
};

} // namespace ck_tile
