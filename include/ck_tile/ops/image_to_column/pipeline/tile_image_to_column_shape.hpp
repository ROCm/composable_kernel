// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
template <typename ThreadTile, // Sequence<...
          typename WarpTile,   // Sequence<...
          typename BlockTile>  // Sequence<...
struct TileImageToColumnShape
{
    static constexpr index_t kMPerThread = ThreadTile::at(number<0>{});
    static constexpr index_t kKPerThread = ThreadTile::at(number<1>{});

    static constexpr index_t kMPerWarp = WarpTile::at(number<0>{});
    static constexpr index_t kKPerWarp = WarpTile::at(number<1>{});

    static constexpr index_t kMThreadPerWarp = kMPerWarp / kMPerThread;
    static constexpr index_t kKThreadPerWarp = kKPerWarp / kKPerThread;

    static constexpr index_t kMPerBlock = BlockTile::at(number<0>{});
    static constexpr index_t kKPerBlock = BlockTile::at(number<1>{});

    static constexpr index_t kMWarpPerBlock = kMPerBlock / kMPerWarp;
    static constexpr index_t kKWarpPerBlock = kKPerBlock / kKPerWarp;

    static constexpr index_t kBlockSize = warpSize * kMWarpPerBlock * kKWarpPerBlock;
};

} // namespace ck_tile
