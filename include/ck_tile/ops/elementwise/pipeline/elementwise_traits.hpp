// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename BlockWarps, typename BlockTile, typename WarpTile, typename Vector>
struct ElementWiseTraits
{
    // TODO: check naming convention
    static constexpr index_t Block_M = BlockTile::at(number<0>{});

    static constexpr index_t Warp_M = WarpTile::at(number<0>{});

    static constexpr index_t Vector_M = Vector::at(number<0>{});

    static constexpr index_t WarpPerBlock_M = BlockWarps::at(number<0>{});

    static constexpr index_t ThreadPerWarp_M = Warp_M / Vector_M;

    static constexpr index_t Repeat_M =
        Block_M /
        (WarpPerBlock_M * Warp_M); // Number of times the warp tile is repeated in the block tile

    static constexpr index_t BlockSize =
        warpSize * reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});
};

} // namespace ck_tile
