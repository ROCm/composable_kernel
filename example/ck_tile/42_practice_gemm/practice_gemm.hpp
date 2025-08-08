// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "practice_gemm_pipeline.hpp"
#include "practice_gemm_block_pipeline_agmem_bgmem_creg.hpp"

namespace ck_tile {

// Problem: defines the nature of the data and the function to apply to the result
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_,
          typename Shape_>
struct PracticeGemmProblem
{
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using CDataType   = CDataType_;
    using AccDataType = AccDataType_;
    using Shape       = remove_cvref_t<Shape_>;
};

template <typename BlockTile_, typename WaveTile_>
struct PracticeGemmShape
{
    using BlockTile = remove_cvref_t<BlockTile_>;
    using WaveTile  = remove_cvref_t<WaveTile_>;

    static constexpr index_t BlockTile_M = BlockTile::at(number<0>{});
    static constexpr index_t BlockTile_N = BlockTile::at(number<1>{});
    static constexpr index_t BlockTile_K = BlockTile::at(number<2>{});

    static constexpr index_t WaveTile_M = WaveTile::at(number<0>{});
    static constexpr index_t WaveTile_N = WaveTile::at(number<1>{});
    static constexpr index_t WaveTile_K = WaveTile::at(number<2>{});

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        return concat('_', "practice_gemm_shape",
                      concat('x', BlockTile_M, BlockTile_N, BlockTile_K),
                      concat('x', WaveTile_M, WaveTile_N, WaveTile_K));
        // clang-format on
    }
};

struct PracticeGemmPolicy
{
    CK_TILE_HOST_DEVICE static constexpr auto MakeBlock2TileMap(index_t M0, index_t N0)
    {
        const auto unmerge = make_merge_transform(make_tuple(N0, M0));

        return [unmerge](index_t block_id) {
            multi_index<2> unmerged;
            unmerge.calculate_lower_index(unmerged, make_multi_index(block_id));

            return make_multi_index(unmerged.at(number<1>{}), unmerged.at(number<0>{}));
        };
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPracticeGemmBlockPipeline()
    {
        using PracticeGemmBlockPipelineProblem_ =
            PracticeGemmBlockPipelineProblem<typename Problem::ADataType,
                                             typename Problem::BDataType,
                                             typename Problem::CDataType,
                                             typename Problem::AccDataType,
                                             typename Problem::Shape>;
        return PracticeGemmBlockPipelineAGmemBGmemCreg<PracticeGemmBlockPipelineProblem_>{};
    }
};

template <typename Problem_, typename Policy_>
struct PracticeGemmKernel
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    CK_TILE_DEVICE void operator()(const typename Problem::ADataType* p_a,
                                   const typename Problem::BDataType* p_b,
                                   typename Problem::CDataType* p_c,
                                   const index_t M,
                                   const index_t N,
                                   const index_t K,
                                   const index_t stride_a,
                                   const index_t stride_b,
                                   const index_t stride_c) const
    {
        const auto a_dram = [&] {
            return make_naive_tensor_view<address_space_enum::global>(
                p_a, make_tuple(M, K), make_tuple(stride_a, 1), number<8>{}, number<1>{});
        }();

        const auto b_dram = [&] {
            return make_naive_tensor_view<address_space_enum::global>(
                p_b, make_tuple(N, K), make_tuple(stride_b, 1), number<8>{}, number<1>{});
        }();

        const auto c_dram = [&] {
            return make_naive_tensor_view<address_space_enum::global>(
                p_c, make_tuple(M, N), make_tuple(stride_c, 1), number<8>{}, number<1>{});
        }();

        PracticeGemmPipeline<Problem, Policy>{}(a_dram, b_dram, c_dram);
    }
};

} // namespace ck_tile
