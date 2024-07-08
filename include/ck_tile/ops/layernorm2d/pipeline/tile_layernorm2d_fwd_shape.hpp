// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
template <typename ThreadTile, // Sequence<...
          typename WarpTile,   // Sequence<...
          typename BlockTile>  // Sequence<...
struct TileLayernorm2dShape
{
    static constexpr index_t kMPerThread = ThreadTile::at(number<0>{});
    static constexpr index_t kNPerThread = ThreadTile::at(number<1>{});

    static constexpr index_t kMPerWarp = WarpTile::at(number<0>{});
    static constexpr index_t kNPerWarp = WarpTile::at(number<1>{});

    static constexpr index_t kMThreadPerWarp = kMPerWarp / kMPerThread;
    static constexpr index_t kNThreadPerWarp = kNPerWarp / kNPerThread;

    static constexpr index_t kMPerBlock = BlockTile::at(number<0>{});
    static constexpr index_t kNPerBlock = BlockTile::at(number<1>{});

    static constexpr index_t kMWarpPerBlock = kMPerBlock / kMPerWarp;
    static constexpr index_t kNWarpPerBlock = kNPerBlock / kNPerWarp;

    // TODO - kNNumWarps can only be 1 if we don't support cross warp welford
    static_assert(kNWarpPerBlock == 1);

    static constexpr index_t kBlockSize = warpSize * kMWarpPerBlock * kNWarpPerBlock;
};

} // namespace ck_tile
