// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

// Traits for batchnorm forward pass configuration
// These are compile-time flags that create different kernel variants
template <bool kSaveMeanInvStd_,
          bool kUpdateMovingAverage_>
struct BatchnormFwdTraits
{
    static constexpr bool kSaveMeanInvStd = kSaveMeanInvStd_;
    static constexpr bool kUpdateMovingAverage = kUpdateMovingAverage_;
};

} // namespace ck_tile
