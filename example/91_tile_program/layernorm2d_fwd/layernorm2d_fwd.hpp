// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "layernorm2d_fwd_kernel.hpp"

template <typename XYDataType,
          typename GammaBetaDataType  = XYDataType,
          typename MeanInvStdDataType = XYDataType>
struct Layernorm2dFwdTypeConfig;

template <>
struct Layernorm2dFwdTypeConfig<ck::half_t>
{
    using XDataType       = ck::half_t;
    using YDataType       = ck::half_t;
    using GammaDataType   = ck::half_t;
    using BetaDataType    = ck::half_t;
    using MeanDataType    = ck::half_t;
    using InvStdDataType  = ck::half_t;
    using ComputeDataType = float;
};

template <>
struct Layernorm2dFwdTypeConfig<ck::half_t, float, float>
{
    using XDataType       = ck::half_t;
    using YDataType       = ck::half_t;
    using GammaDataType   = float;
    using BetaDataType    = float;
    using MeanDataType    = float;
    using InvStdDataType  = float;
    using ComputeDataType = float;
};
