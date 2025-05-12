// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v2.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using F16 = ck::half_t;
using F32 = float;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmInstance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV2<
        ALayout,   BLayout,  CLayout,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        2,   256,
        256, 256, 
        32, 8, 4,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>,
        1, 8, 4, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>;
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

struct GemmTypeConifg_RR
{
    using ADataType_ = ck::half_t;
    using BDataType_ = ck::half_t;
    using CDataType_ = ck::half_t;
    using ALayout_   = Row;
    using BLayout_   = Row;
    using CLayout_   = Row;
};

struct GemmTypeConifg_RC
{
    using ADataType_ = ck::half_t;
    using BDataType_ = ck::half_t;
    using CDataType_ = ck::half_t;
    using ALayout_   = Row;
    using BLayout_   = Col;
    using CLayout_   = Row;
};

using DeviceGemmInstance_0 = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV2<
        ALayout,   BLayout,  CLayout,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        2,   256,
        256, 256, 
        32, 8, 4,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>,
        1, 8, 4, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>;

using DeviceGemmInstance_1 =
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV2<
        ALayout,   BLayout,  CLayout,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        2,   256,
        256, 256, 
        32, 8, 8,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<4, 32, 2>,  S<0, 2, 1>,  S<0, 2, 1>,
        1, 8, 8, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>;

using DeviceGemmInstance_2 =
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV2<
        ALayout,   BLayout,  CLayout,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        2,   256,
        256, 256, 
        32, 8, 8,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>;

using DeviceGemmInstance_3 =
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV2<
        ALayout,   BLayout,  CLayout,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        2,   256,
        256, 256, 
        32, 8, 4,
        32,   32,
        4,    4, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 4, 8, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>;

using DeviceGemmInstance_4 =
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV2<
        ALayout,   BLayout,  CLayout,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        2,   256,
        256, 256, 
        32, 8, 8,
        16,   16,
        8,    8, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<4, 32, 2>,  S<0, 2, 1>,  S<0, 2, 1>,
        1, 8, 8, 0,
        1, 1, S<1, 32, 1, 8>, 4,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>;

using DeviceGemmInstance_5 =
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV2<
        ALayout,   BLayout,  CLayout,   
        F16,   F16,  F16,  F32,  F16, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        2,   256,
        256, 256, 
        32, 8, 4,
        16,   16,
        8,    8, 
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 4, 8, 0,
        1, 1, S<1, 32, 1, 8>, 4,
        ck::LoopScheduler::Default, ck::PipelineVersion::v1>;


int main(int argc, char* argv[])
{
   run_gemm_example_with_instance<DeviceGemmInstance_0, GemmTypeConifg_RR>(argc, argv);
   run_gemm_example_with_instance<DeviceGemmInstance_1, GemmTypeConifg_RR>(argc, argv);
   run_gemm_example_with_instance<DeviceGemmInstance_4, GemmTypeConifg_RR>(argc, argv);

   run_gemm_example_with_instance<DeviceGemmInstance_2, GemmTypeConifg_RC>(argc, argv);
   run_gemm_example_with_instance<DeviceGemmInstance_3, GemmTypeConifg_RC>(argc, argv);
   run_gemm_example_with_instance<DeviceGemmInstance_5, GemmTypeConifg_RC>(argc, argv);
}
