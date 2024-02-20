// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tile_program {

template <index_t kMPerBlock_, index_t kNPerBlock_>
struct TileLayernorm2dShape
{
    // TODO: Extract StaticTileDistributionEncoding into here
    static constexpr index_t kM = kMPerBlock_;
    static constexpr index_t kN = kNPerBlock_;
};

} // namespace tile_program
} // namespace ck
