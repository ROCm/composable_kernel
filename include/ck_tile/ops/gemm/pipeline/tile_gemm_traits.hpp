// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

template <bool kPadA_,
          bool kPadB_,
          bool kPadC_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_>
struct TileGemmTraits
{
    static constexpr bool kPadA = kPadA_;
    static constexpr bool kPadB = kPadB_;
    static constexpr bool kPadC = kPadC_;

    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;
};

} // namespace ck_tile
