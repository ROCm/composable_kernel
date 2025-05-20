// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/norm_reduce/block/block_norm_reduce_problem.hpp"
#include "ck_tile/ops/norm_reduce/block/block_norm_reduce.hpp"

namespace ck_tile {
struct ElementWiseDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>, // Replicate
                                       tuple<sequence<S::Repeat_M,
                                                      S::WarpPerBlock_M,
                                                      S::ThreadPerWarp_M,
                                                      S::Vector_M>>,    // Hierarchical
                                       tuple<sequence<1>, sequence<1>>, // Parallel
                                       tuple<sequence<1>, sequence<2>>, // Parallel
                                       sequence<1, 1>,                  // Yield
                                       sequence<0, 3>>{}                // Yield
        );
    }
};

}
