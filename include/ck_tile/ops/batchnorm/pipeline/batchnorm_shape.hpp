// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// BatchnormShape defines the block/warp/thread tile configuration
// Similar to reduce2d_shape but adapted for batchnorm
template <typename BlockWarps_,
          typename BlockTile_,
          typename WarpTile_,
          typename Vector_>
struct BatchnormShape
{
    using BlockWarps = remove_cvref_t<BlockWarps_>;
    using BlockTile  = remove_cvref_t<BlockTile_>;
    using WarpTile   = remove_cvref_t<WarpTile_>;
    using Vector     = remove_cvref_t<Vector_>;

    static constexpr index_t kBlockWarps_M = BlockWarps::at(number<0>{});
    static constexpr index_t kBlockWarps_N = BlockWarps::at(number<1>{});

    static constexpr index_t kBlockSize = kBlockWarps_M * kBlockWarps_N * get_warp_size();

    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t WarpTile_M = WarpTile::at(number<0>{});
    static constexpr index_t WarpTile_N = WarpTile::at(number<1>{});

    static constexpr index_t VectorSize_M = Vector::at(number<0>{});
    static constexpr index_t VectorSize_N = Vector::at(number<1>{});

    // Thread tile sizes
    static constexpr index_t ThreadTile_M = WarpTile_M / kBlockWarps_M;
    static constexpr index_t ThreadTile_N = WarpTile_N / kBlockWarps_N;

    // For batchnorm:
    // - M dimension represents the batch*channel (merged N*C)
    // - N dimension represents the spatial (merged H*W)
    // We reduce over N (spatial) dimension per M (batch*channel)
    
    static constexpr auto BlockSize = number<kBlockSize>{};
};

} // namespace ck_tile
