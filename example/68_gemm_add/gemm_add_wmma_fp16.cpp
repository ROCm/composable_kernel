// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = F16;
using EDataType        = F16;

using ALayout = Row;
using BLayout = Col;
using DLayout = Row;
using ELayout = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Add;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffle<
    ALayout,
    BLayout,
    ck::Tuple<DLayout>,
    ELayout,
    ADataType,
    BDataType,
    AccDataType,
    CShuffleDataType,
    ck::Tuple<DDataType>,
    EDataType,
    AElementOp,
    BElementOp,
    CDEElementOp,
    GemmSpec,
    2,   // Prefetch stage
    128, // BlockSize
    128, // MPerBlock
    64,  // NPerBlock
    64,  // KPerBlock
    8,   // K1
    16,  // MPerWmma
    16,  // NPerWmma
    4,   // M-Repeat // M-PerWmma / M-Repeat = M-Wave
    2,   // N-Repeat // N-PerWmma / N-Repeat = N-Wave
    S<4, 32, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    true,
    S<4, 32, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    true,
    1, // C shuffle (M Repeat) Per store
    1, // C shuffle (N Repeat) Per store
    S<1, 32, 1, 4>,
    8>;

// clang-format on

#include "run_gem_add_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_add_example(argc, argv); }
