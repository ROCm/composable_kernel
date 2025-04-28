// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          bool IsInputGemm_,
          bool IsGateOnly_,
          bool IsFusedQuant_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          typename GateActivation_>
struct TileMoeGemmTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    static constexpr bool IsInputGemm  = IsInputGemm_;
    static constexpr bool IsGateOnly   = IsGateOnly_;
    static constexpr bool IsFusedQuant = IsFusedQuant_;

    // TODO this can't be hardcoded here! Should be in policy!
    static constexpr int _VectorSize = 16;

    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;

    using GateActivation = remove_cvref_t<GateActivation_>;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;
};


} // namespace ck_tile
