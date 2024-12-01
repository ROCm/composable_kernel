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
          bool HasHotLoop_,
          index_t TailNum_>
struct TileGemmTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    static constexpr int _VectorSize = 16;

    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;
    static constexpr bool HasHotLoop = HasHotLoop_;
    static constexpr auto TailNum    = TailNum_;
};

} // namespace ck_tile
