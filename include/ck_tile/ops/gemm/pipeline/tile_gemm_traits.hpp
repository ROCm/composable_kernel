// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_>
struct TileGemmTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    // TODO this can't be hardcoded here! Should be in policy!
    static constexpr int _VectorSize = 16;

    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;
};

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          bool DoubleSmemBuffer_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          bool TransposeC_            = false,
          bool UseStructuredSparsity_ = false>
struct TileGemmUniversalTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    static constexpr bool DoubleSmemBuffer = DoubleSmemBuffer_;

    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;

    static constexpr bool TransposeC            = TransposeC_;
    static constexpr bool UseStructuredSparsity = UseStructuredSparsity_;
};

} // namespace ck_tile
