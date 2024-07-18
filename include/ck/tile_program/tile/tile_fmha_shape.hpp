// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tile_program {

template <index_t kM0PerTile_, // tile size along q seqlen
          index_t kN0PerTile_, // tile size along k seqlen
          index_t kK0PerTile_, // tile size along qk gemm unroll
          index_t kN1PerTile_, // tile size along v head_dim
          index_t kK1PerTile_  // tile size along kv gemm unroll
          >
struct TileFmhaShape
{
    static constexpr index_t kM0 = kM0PerTile_;
    static constexpr index_t kN0 = kN0PerTile_;
    static constexpr index_t kK0 = kK0PerTile_;
    static constexpr index_t kN1 = kN1PerTile_;
    static constexpr index_t kK1 = kK1PerTile_;
};

} // namespace tile_program
} // namespace ck
