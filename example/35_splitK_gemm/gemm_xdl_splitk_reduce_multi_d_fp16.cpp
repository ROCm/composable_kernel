// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3r1.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;
using ReduceDataType   = float;
using D0DataType       = ck::half_t;
using DsDataType       = ck::Tuple<D0DataType>;

using ALayout  = Row;
using BLayout  = Row;
using CLayout  = Row;
using D0Layout = CLayout;
using DsLayout = ck::Tuple<D0Layout>;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Add;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// clang-format off
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3R1<
        ALayout,   BLayout,   DsLayout,  CLayout,
        ADataType,   BDataType, DsDataType,  CDataType, AccDataType,  CShuffleDataType,
	AElementOp, BElementOp, CDEElementOp, GemmDefault, 
        256,
        128,  128,  64,
        8,    4,
        32,   32,
        2,    2,
        S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,
        2,    8,    8,   0,
        S<16, 16, 1>,    S<0, 2, 1>,    S<0, 2, 1>,
        1,    8,    4,   0,
        1,    1,    S<1, 32, 1, 8>,  8,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v2, ReduceDataType>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        PassThrough>;

#include "run_gemm_splitk_reduce_multi_d_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
