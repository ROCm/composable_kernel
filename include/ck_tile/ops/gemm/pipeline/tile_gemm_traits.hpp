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
          typename CLayout_,
          index_t NumWaveGroups_ = 1>
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
    static constexpr index_t NumWaveGroups      = NumWaveGroups_;
};

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          bool DoubleSmemBuffer_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          bool TransposeC_            = false,
          bool UseStructuredSparsity_ = false,
          bool UsePersistentKernel_   = false,
          index_t NumWaveGroups_      = 1,
          bool Preshuffle_            = 0>
struct TileGemmUniversalTraits
{
    static constexpr bool kPadM            = kPadM_;
    static constexpr bool kPadN            = kPadN_;
    static constexpr bool kPadK            = kPadK_;
    static constexpr int _VectorSize       = 16;
    static constexpr bool DoubleSmemBuffer = DoubleSmemBuffer_;

    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;

    static constexpr bool TransposeC            = TransposeC_;
    static constexpr bool UseStructuredSparsity = UseStructuredSparsity_;
    static constexpr bool UsePersistentKernel   = UsePersistentKernel_;
    static constexpr index_t NumWaveGroups      = NumWaveGroups_;
    static constexpr bool Preshuffle            = Preshuffle_;
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
using PersistentTileGemmUniversalTraits = TileGemmUniversalTraits<kPadM_,
                                                                  kPadN_,
                                                                  kPadK_,
                                                                  DoubleSmemBuffer_,
                                                                  ALayout_,
                                                                  BLayout_,
                                                                  CLayout_,
                                                                  TransposeC_,
                                                                  UseStructuredSparsity_,
                                                                  true>;

} // namespace ck_tile
