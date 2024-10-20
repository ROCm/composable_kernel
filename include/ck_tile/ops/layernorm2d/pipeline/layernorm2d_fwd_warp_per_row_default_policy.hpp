// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/welford/block/block_welford_problem.hpp"
#include "ck_tile/ops/welford/block/block_welford.hpp"

namespace ck_tile {

struct Layernorm2dFwdWarpPerRowDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::BlockWarps_M, S::Thread_M, S::Vector_M>,
                      sequence<S::Repeat_N, S::BlockWarps_N, S::Thread_N, S::Vector_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<0, 1>, sequence<1, 2>>,
                sequence<1, 2, 2>,
                sequence<2, 0, 3>>{});
    }
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeGammaBetaBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<S::BlockWarps_M, S::Thread_M>,
                tuple<sequence<S::Repeat_N, S::BlockWarps_N, S::Thread_N, S::Vector_N>>,
                tuple<sequence<0, 1>, sequence<0, 1>>,
                tuple<sequence<0, 1>, sequence<1, 2>>,
                sequence<1, 1>,
                sequence<0, 3>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockWelford()
    {
        using P_ = BlockWelfordProblem<typename Problem::XDataType,
                                       typename Problem::ComputeDataType,
                                       typename Problem::BlockShape>;

        return BlockWelford<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockWelfordSync()
    {
        using P_ = BlockWelfordProblem<typename Problem::XDataType,
                                       typename Problem::ComputeDataType,
                                       typename Problem::BlockShape>;

        return BlockWelfordSync<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockWelfordCrossWarpSync()
    {
        using P_ = BlockWelfordProblem<typename Problem::XDataType,
                                       typename Problem::ComputeDataType,
                                       typename Problem::BlockShape>;

        return BlockWelfordCrossWarpSync<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        if constexpr(Problem::kNeedCrossWarpSync)
        {
            using P_ = BlockWelfordProblem<typename Problem::XDataType,
                                           typename Problem::ComputeDataType,
                                           typename Problem::BlockShape>;

            using block_welford = BlockWelford<P_>;
            using x_block_tile =
                decltype(make_static_distributed_tensor<typename Problem::XDataType>(
                    MakeXBlockTileDistribution<Problem>()));
            using mean_var_block_tile =
                decltype(block_welford::template MakeMeanVarBlockTile<x_block_tile>());

            return GetBlockWelfordCrossWarpSync<Problem>()
                .template GetSmemSize<mean_var_block_tile>();
        }
        else
        {
            return 1; // zero size arrays are an extension
        }
    }
};
} // namespace ck_tile
