// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
template <typename BlockTile_, typename BlockWarps_, typename WarpTile_, typename Vector>
struct TileReduceShape
{
    using BlockTile  = remove_cvref_t<BlockTile_>;
    using BlockWarps = remove_cvref_t<BlockWarps_>;
    using WarpTile   = remove_cvref_t<WarpTile_>;

    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t WarpPerBlock_M = BlockWarps::at(number<0>{});
    static constexpr index_t WarpPerBlock_N = BlockWarps::at(number<1>{});

    static constexpr index_t Warp_M = WarpTile::at(number<0>{});
    static constexpr index_t Warp_N = WarpTile::at(number<1>{});

    static constexpr index_t Vector_M = Vector::at(number<0>{});
    static constexpr index_t Vector_N = Vector::at(number<1>{});

    static constexpr index_t MPerWarp = Block_M / WarpPerBlock_M;
    static constexpr index_t NPerWarp = Block_N / WarpPerBlock_N;

    static constexpr index_t ThreadTile_M = MPerWarp / Vector_M;
    static constexpr index_t ThreadTile_N = NPerWarp / Vector_N;

    static constexpr index_t MThreadPerWarp = MPerWarp / ThreadTile_M;
    static constexpr index_t NThreadPerWarp = NPerWarp / ThreadTile_N;

    static constexpr index_t NumWarps = reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});
};
} // namespace ck_tile
