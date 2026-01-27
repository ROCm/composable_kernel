// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Simple GEMM shape for MHC operations
// This provides the kM, kN, kK members that BlockGemm expects
template <index_t M_, index_t N_, index_t K_>
struct MHCGemmShape
{
    static constexpr index_t kM = M_;
    static constexpr index_t kN = N_;
    static constexpr index_t kK = K_;
    
    // For compatibility with BlockGemm
    static constexpr index_t NumWarps = 1;  // Simple: 1 warp for now
    static constexpr index_t kBlockSize = 256;  // Block size
};

} // namespace ck_tile
