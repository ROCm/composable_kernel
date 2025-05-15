// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp"

using ADataType        = ck::f8_t;
using BDataType        = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
using CDataType        = ck::bhalf_t;
using ComputeTypeA     = ck::f8_t;
using ComputeTypeB     = ck::f8_t;

using ALayout = Col;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmV2Instance = ck::tensor_operation::device::DeviceGemm_Wmma_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        128,
        64, 64, 
        32, 8, 8,
        16,   16,
        2,    2, 
        S<4, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<4, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 2, 4, 1, 1, 1,
        S<1, 32, 1, 2>,   8,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3, ComputeTypeA, ComputeTypeB>;
// clang-format on

using ReferenceComputeType  = ck::f8_t;
using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CElementOp,
                                                                        ReferenceComputeType,
                                                                        ReferenceComputeType>;

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
