// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include <cstdint>

namespace ck_tile {

enum struct QuantType : std::uint16_t
{
    AQuantGrouped = 0,
    BQuantGrouped = 1,
    RowColQuant   = 2
};

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          bool PreshuffleQuant_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          QuantType QuantType_,
          typename AQLayout_        = ALayout_,
          typename BQLayout_        = BLayout_,
          bool DoubleSmemBuffer_    = false,
          bool UsePersistentKernel_ = false>
struct TileGemmQuantTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    static constexpr QuantType kQuantType = QuantType_;

    static constexpr int _VectorSize       = 16;
    static constexpr bool DoubleSmemBuffer = DoubleSmemBuffer_;

    using ALayout  = ALayout_;
    using BLayout  = BLayout_;
    using CLayout  = CLayout_;
    using AQLayout = AQLayout_;
    using BQLayout = BQLayout_;

    // TODO: It should be replaced to single value
    using AsLayout = ALayout_;
    using BsLayout = BLayout_;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;
    static constexpr index_t NumWaveGroups      = 1;
    static constexpr bool UsePersistentKernel   = UsePersistentKernel_;

    static constexpr bool PreshuffleQuant = PreshuffleQuant_;
};

} // namespace ck_tile
