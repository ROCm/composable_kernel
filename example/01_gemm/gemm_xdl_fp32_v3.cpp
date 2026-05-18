// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#define CK_ENABLE_DYNAMIC_WARP_SIZE 1
#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

using ADataType        = float;
using BDataType        = float;
using AccDataType      = float;
using CShuffleDataType = float;
using CDataType        = float;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        128,
        64, 64, 
        64, 4, 4,
        16,   16,
        2,    4,
        S<4, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 2, 2, 0,
        S<4, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 2, 2, 0,
        1, 2, S<1, 32, 1, 4>, 2,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
