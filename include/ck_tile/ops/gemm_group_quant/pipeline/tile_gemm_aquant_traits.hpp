// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          typename AQLayout_ = ALayout_>
struct TileGemmAQuantTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    static constexpr int _VectorSize = 16;

    using ALayout  = ALayout_;
    using BLayout  = BLayout_;
    using CLayout  = CLayout_;
    using AQLayout = AQLayout_;

    static constexpr bool UseStructuredSparsity = false;
    static constexpr index_t NumWaveGroups      = 1;
};

} // namespace ck_tile
