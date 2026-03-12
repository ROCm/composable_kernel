// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_b_preshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/amd_ck_fp8.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/get_id.hpp"
#include "ck/utility/scheduler_enum.hpp"

#include <cstddef>
#include <iostream>
#include <type_traits>

using F8  = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;

using ADataType        = F8;
using BDataType        = F8;
using AccDataType      = F32;
using CShuffleDataType = F32;
using CDataType        = F16;
using ComputeTypeA     = F8;
using ComputeTypeB     = F8;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr bool PermuteA = false;
static constexpr bool PermuteB = false;
static constexpr int KPack     = 16; // int4 -> 32, fp8 -> 16, fp16 -> 8
// clang-format off
using DeviceOpInstance = 
    ck::tensor_operation::device::DeviceGemm_Wmma_CShuffleV3_BPreshuffle<
        ALayout,   BLayout,  CLayout,   
        ADataType, BDataType, CDataType, AccDataType, CShuffleDataType, 
        AElementOp, BElementOp, CElementOp, GemmDefault, 
        256,
        32, 128, 256,
        16, 16,
        16, 16,
        2, 1,
        S<16, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 16, 16, 0,
        S<16, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 16, 16, 0,
        1, 1, S<1, 16, 1, 16>, S<8, 8, 1>,
        ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, ComputeTypeA, ComputeTypeB, PermuteA, PermuteB>;
// clang-format on

#include "run_gemm_wmma_bpreshuffle_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
