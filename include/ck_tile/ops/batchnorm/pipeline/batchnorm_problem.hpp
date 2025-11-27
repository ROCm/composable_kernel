// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batchnorm/pipeline/batchnorm_shape.hpp"
#include "ck_tile/ops/batchnorm/pipeline/batchnorm_fwd_traits.hpp"

namespace ck_tile {

// BatchnormProblem defines the computational problem for batch normalization
// Input: x with shape [N, C, H, W]
// Output: y with shape [N, C, H, W]
// Reduction over batch (N) and spatial dimensions (H, W) per channel (C)
template <typename XDataType_,
          typename GammaDataType_,
          typename BetaDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename MeanVarDataType_,
          typename BlockShape_,
          typename Traits_>
struct BatchnormProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using GammaDataType   = remove_cvref_t<GammaDataType_>;
    using BetaDataType    = remove_cvref_t<BetaDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using MeanVarDataType = remove_cvref_t<MeanVarDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
    using Traits          = remove_cvref_t<Traits_>;
};

} // namespace ck_tile
