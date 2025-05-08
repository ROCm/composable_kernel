// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp"

using F8   = f8_t;
using BF16 = bhalf_t;
using F32  = float;

using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

using PassThrough = element_wise::PassThrough;

using ADataType        = F8;
using BDataType        = F8;
using AccDataType      = F32;
using CShuffleDataType = BF16;
using CDataType        = BF16;

using ALayout = Col;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::GemmSpec;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Wmma_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmSpec, 
        32,
        16, 16, 
        64, 8, 8,
        16,   16,
        1,    1, 
        S<2, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 1,
        S<2, 16, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 1, 4, 0, 1, 1,
        S<1, 16, 1, 2>,   8,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

using ReferenceGemmInstanceGPU = ck::tensor_operation::device::ReferenceGemm<ALayout,
                                                                             BLayout,
                                                                             CLayout,
                                                                             ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             AccDataType,
                                                                             AElementOp,
                                                                             BElementOp,
                                                                             CElementOp>;

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
