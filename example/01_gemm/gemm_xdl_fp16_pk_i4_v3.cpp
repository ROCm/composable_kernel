// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::pk_i4_t;
using AccDataType      = float;
using CShuffleDataType = float;
using CDataType        = ck::half_t;

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
        ADataType, BDataType, CDataType, AccDataType, CShuffleDataType, 
        AElementOp, BElementOp, CElementOp, GemmDefault, 
#if 0
        64,
        16, 16, 
        256, 8, 32,
        16,   16,
        1,    1, 
        S<32, 2, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8,  8, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 32, 32, 0,
        1, 1, S<1, 16, 1, 4>, 4,
        ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v2>;
#else
        128,
        16, 32, 
        128, 8, 32,
        16,   16,
        1,    1, 
        S<16, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        S<4, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 32, 32, 0,
        1, 1, S<1, 16, 1, 8>, 4,
        ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1>;

#endif
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        PassThrough>;

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
