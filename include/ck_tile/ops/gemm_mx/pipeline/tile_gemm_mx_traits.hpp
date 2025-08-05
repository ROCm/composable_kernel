// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          typename ALayout_,
          typename AScaleLayout_,
          typename BLayout_,
          typename BScaleLayout_,
          typename CLayout_>
struct TileGemmMXTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    static constexpr int _VectorSize = 16;

    using ALayout      = ALayout_;
    using AScaleLayout = AScaleLayout_;
    using BLayout      = BLayout_;
    using BScaleLayout = BScaleLayout_;
    using CLayout      = CLayout_;

    static constexpr bool UseStructuredSparsity = false;
    static constexpr index_t NumWaveGroups      = 1;
};

} // namespace ck_tile
