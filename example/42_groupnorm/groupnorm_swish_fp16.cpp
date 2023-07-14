// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

constexpr int Rank         = 5;
constexpr int NumReduceDim = 3;

using XDataType       = ck::half_t;
using GammaDataType   = ck::half_t;
using BetaDataType    = ck::half_t;
using YDataType       = ck::half_t;
using ComputeDataType = float;
using YElementOp      = ck::tensor_operation::element_wise::Swish;

using DeviceInstance =
    ck::tensor_operation::device::DeviceNormalizationImpl<XDataType,
                                                          GammaDataType,
                                                          BetaDataType,
                                                          ComputeDataType,
                                                          YDataType,
                                                          YElementOp,
                                                          Rank,
                                                          NumReduceDim,
                                                          1024, // BlockSize
                                                          1,    // ClusterM
                                                          1024, // ClusterK
                                                          1,    // SliceM
                                                          32,   // SliceK
                                                          1,    // SrcVecDim (0=M, 1=K)
                                                          2,    // SrcScalarPerVector
                                                          1,    // GammaVecDim (0=M, 1=K)
                                                          2,    // GammaScalarPerVector
                                                          1,    // BetaVecDim (0=M, 1=K)
                                                          2,    // BetaScalarPerVector
                                                          2>;   // OutScalarPerVector

#include "run_groupnorm_example.inc"

int main(int argc, char* argv[]) { run_groupnorm_example(argc, argv); }
