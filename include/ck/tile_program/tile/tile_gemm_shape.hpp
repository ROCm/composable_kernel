// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tile_program {

template <index_t kMPerTile, index_t kNPerTile, index_t kKPerTile>
struct TileGemmShape
{
    static constexpr index_t kM = kMPerTile;
    static constexpr index_t kN = kNPerTile;
    static constexpr index_t kK = kKPerTile;
};

} // namespace tile_program
} // namespace ck
