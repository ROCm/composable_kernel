// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadA_,
          bool kPadB_,
          bool kPadC_,
          typename LayoutA_,
          typename LayoutB_,
          typename LayoutC_>
struct TileGemmTraits
{
    static constexpr bool kPadA = kPadA_;
    static constexpr bool kPadB = kPadB_;
    static constexpr bool kPadC = kPadC_;

    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
};

} // namespace ck_tile
