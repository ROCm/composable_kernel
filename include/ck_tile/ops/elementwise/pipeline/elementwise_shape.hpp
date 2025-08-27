// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename BlockWarps, typename BlockTile, typename WarpTile, typename ComputeDataType>
struct ElementWiseShape
{
    static constexpr index_t kBlockM = BlockTile::at(number<0>{});

    static constexpr index_t kWarpM = WarpTile::at(number<0>{});

    static constexpr index_t kVectorM = 16 / sizeof(ComputeDataType);

    static constexpr index_t kWarpPerBlockM = BlockWarps::at(number<0>{});

    static constexpr index_t kThreadPerWarpM = kWarpM / kVectorM;

    static constexpr index_t kRepeatM = kBlockM / (kWarpPerBlockM * kWarpM);

    static constexpr index_t kBlockSize =
        ck_tile::get_warp_size() * reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});
};

} // namespace ck_tile
