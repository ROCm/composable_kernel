// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/generic_2d_block_shape.hpp"

namespace ck_tile {

// BatchnormShape using Generic2dBlockShape for proper tile distribution support
template <typename BlockTile_,      // Block size, sequence<M, N>
          typename ThreadPerBlock_, // Threads along sequence<M, N>
          typename Vector_>         // Vector size along sequence<M, N>
using BatchnormShape = Generic2dBlockShape<BlockTile_, ThreadPerBlock_, Vector_>;

} // namespace ck_tile
