// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/norm_reduce/block/block_merge_problem.hpp"
#include "ck_tile/ops/norm_reduce/block/block_merge.hpp"

namespace ck_tile {

struct Layernorm2dFwdPipelineDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
                      sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeGammaBetaBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<S::WarpPerBlock_M, S::ThreadPerWarp_M>,
                tuple<sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<0, 1>, sequence<0, 1>>,
                tuple<sequence<0, 1>, sequence<1, 2>>,
                sequence<1, 1>,
                sequence<0, 3>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockMerge()
    {
        using P_ = BlockMergeProblem<typename Problem::ComputeDataType,
                                     typename Problem::ComputeDataType,
                                     typename Problem::BlockShape,
                                     Problem::Traits::kFastFDiv,
                                     Problem::Traits::kWelford>;
        return BlockMerge<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockMergeSync()
    {
        using P_ = BlockMergeProblem<typename Problem::ComputeDataType,
                                     typename Problem::ComputeDataType,
                                     typename Problem::BlockShape,
                                     Problem::Traits::kFastFDiv,
                                     Problem::Traits::kWelford>;

        return BlockMergeSync<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockMergeCrossWarpSync()
    {
        using P_ = BlockMergeProblem<typename Problem::ComputeDataType,
                                     typename Problem::ComputeDataType,
                                     typename Problem::BlockShape,
                                     Problem::Traits::kFastFDiv,
                                     Problem::Traits::kWelford>;

        return BlockMergeCrossWarpSync<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        if constexpr(Problem::kNeedCrossWarpSync)
        {
            using P_ = BlockMergeProblem<typename Problem::ComputeDataType,
                                         typename Problem::ComputeDataType,
                                         typename Problem::BlockShape,
                                         Problem::Traits::kFastFDiv,
                                         Problem::Traits::kWelford>;

            using block_welford = BlockMerge<P_>;
            using x_block_tile =
                decltype(make_static_distributed_tensor<typename Problem::ComputeDataType>(
                    MakeXBlockTileDistribution<Problem>()));
            using mean_var_block_tile =
                decltype(block_welford::template MakeMeanVarBlockTile<x_block_tile>());

            return GetBlockMergeCrossWarpSync<Problem>()
                .template GetSmemSize<mean_var_block_tile>();
        }
        else
        {
            return 1; // zero size arrays are an extension
        }
    }
};
} // namespace ck_tile
