// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include <type_traits>

namespace ck_tile {

template <typename DataType_,
          typename BlockTile, // Sequence<...
          typename ThreadTile,
          bool kPadM_ = false,
          bool kPadN_ = false> // Sequence<...
struct BatchedTransposeProblem
{
    using DataType = remove_cvref_t<DataType_>;

    static constexpr index_t kMPerThread = ThreadTile::at(number<0>{});
    static constexpr index_t kNPerThread = ThreadTile::at(number<1>{});

    static constexpr index_t kMPerWarp = get_warp_size();
    static constexpr index_t kNPerWarp = get_warp_size();

    static constexpr index_t kMThreadPerWarp = kMPerWarp / kMPerThread;
    static constexpr index_t kNThreadPerWarp = kNPerWarp / kNPerThread;

    static constexpr index_t kMPerBlock = BlockTile::at(number<0>{});
    static constexpr index_t kNPerBlock = BlockTile::at(number<1>{});

    static constexpr index_t kMWarpPerBlock = kMPerBlock / kMPerWarp;
    static constexpr index_t kNWarpPerBlock = kNPerBlock / kNPerWarp;

    static constexpr index_t kBlockSize =
        kMThreadPerWarp * kNThreadPerWarp * kMWarpPerBlock * kNWarpPerBlock;

    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;

    // 128-bit is the max single-instruction bandwidth for load/store
    static constexpr index_t MaxLoadStoreSize = 16;
    static constexpr index_t VectorSizeInput  = kPadM ? 1 : MaxLoadStoreSize / sizeof(DataType);
    static constexpr index_t VectorSizeOutput = kPadN ? 1 : MaxLoadStoreSize / sizeof(DataType);
};
} // namespace ck_tile
