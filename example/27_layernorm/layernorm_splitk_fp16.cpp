// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using XDataType       = ck::half_t;
using GammaDataType   = ck::half_t;
using BetaDataType    = ck::half_t;
using YDataType       = ck::half_t;
using ComputeDataType = float;
using PassThrough     = ck::tensor_operation::element_wise::PassThrough;

constexpr int Rank         = 2;
constexpr int NumReduceDim = 1;

using DeviceInstance =
    ck::tensor_operation::device::DeviceNormalizationSplitKImpl<XDataType,
                                                                GammaDataType,
                                                                BetaDataType,
                                                                ComputeDataType,
                                                                YDataType,
                                                                PassThrough,
                                                                Rank,
                                                                NumReduceDim,
                                                                256, // BlockSize
                                                                8,   // ClusterM
                                                                32,  // ClusterK
                                                                1,   // SliceM
                                                                8,   // SliceK
                                                                1,   // XYVectorDim (0=M, 1=K)
                                                                8,   // XScalarPerVector
                                                                1,   // GammaVecDim (0=M, 1=K)
                                                                8,   // GammaScalarPerVector
                                                                1,   // BetaVecDim (0=M, 1=K)
                                                                8,   // BetaScalarPerVector
                                                                8>;  // YScalarPerVector

#include "run_layernorm_example.inc"

int main() { return run_groupnorm_example<DeviceInstance>(); }
