// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"

namespace ck_tile {

// GEMM shape for MHC operations
// This provides the kM, kN, kK members and warp configuration
template <index_t M_, index_t N_, index_t K_>
using MHCGemmShape =
    TileGemmShape<sequence<M_, N_, K_>,  // BlockTile
                  sequence<1, 1, 1>,     // BlockWarps (1 warp in M, N, K)
                  sequence<M_, N_, K_>>; // WarpTile (same as block tile for single warp)

} // namespace ck_tile
