// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batchnorm/pipeline/batchnorm_shape.hpp"

namespace ck_tile {

// BatchnormProblem defines the computational problem for batch normalization
// Input: x with shape [N, C, H, W]
// Output: y with shape [N, C, H, W]
// Reduction over spatial dimensions (H, W) per channel
template <typename XDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename Shape_>
struct BatchnormProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using BlockShape      = remove_cvref_t<Shape_>;

    // For now, start with simple forward pass without scale/bias
    // We'll add these later:
    // using GammaDataType = ...  // scale parameter
    // using BetaDataType = ...   // bias parameter
    // using MeanVarDataType = ... // for saving mean/variance
};

} // namespace ck_tile
