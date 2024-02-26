// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tile_program {

template <index_t kMPerTile_, index_t kNPerTile_>
struct TileLayernorm2dShape
{
    // TODO: Extract StaticTileDistributionEncoding into here
    static constexpr index_t kM = kMPerTile_;
    static constexpr index_t kN = kNPerTile_;
};

} // namespace tile_program
} // namespace ck
