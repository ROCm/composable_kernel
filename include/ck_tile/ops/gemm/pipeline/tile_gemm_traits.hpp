// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          bool kBlockDefaultPolicy_,
          int kBlockMethod_,
          int kBlockPolicyMethod_>
struct TileGemmTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    static constexpr bool kBlockDefaultPolicy = kBlockDefaultPolicy_;
    static constexpr int kBlockMethod         = kBlockMethod_;
    static constexpr int kBlockPolicyMethod   = kBlockPolicyMethod_;

    static constexpr int _VectorSize = 16;

    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;
};

} // namespace ck_tile
