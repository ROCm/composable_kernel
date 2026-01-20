// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp"

namespace ck_tile {

struct SinkhornKnoppDefaultPolicy : public Reduce2dDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeTransposedXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::ThreadTile_N>,
                    sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::ThreadTile_M>>,
                tuple<sequence<2, 1>, sequence<2, 1>>,
                tuple<sequence<1, 1>, sequence<2, 1>>, // WarpPerBlock_M, WarpPerBlock_N;
                                                       // ThreadPerWarp_M, ThreadPerWarp_N
                // sequence<1, 1, 2, 2>,
                // sequence<0, 3, 0, 3>>{}); // Repeat_N, ThreadTile_N, Repeat_M, ThreadTile_M
                sequence<2, 2, 1, 1>,
                sequence<0, 2, 0, 3>>{}); // Repeat_M, ThreadTile_M, Repeat_N, ThreadTile_N
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::ThreadTile_M>,
                    sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::ThreadTile_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<1, 1>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }
};

} // namespace ck_tile
