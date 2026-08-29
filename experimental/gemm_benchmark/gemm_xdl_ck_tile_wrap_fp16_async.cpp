// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include "common.hpp"

#include "gemm_xdl_ck_tile_wrap.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

using GemmDefault = ck_tile::sequence<false, false, false>; // M/N/K Pad

// clang-format off
using DeviceGemmV2Instance = ck::tensor_operation::device::DeviceGemm_Xdl_CkTileWrap<
    ALayout, BLayout, CLayout,
    ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,
    AElementOp, BElementOp, CElementOp,
    GemmDefault,
    256, 256, 64,                                      // M/N/K PerBlock
    16, 16, ck_tile::get_k_warp_tile<ck_tile::fp16_t, 16>(),  // M/N/K PerXDL
    2, 2, 1, 1,                                        // M/N/K Warp
    ADataType,
    1,
    1,
    ck_tile::GemmPipelineScheduler::Intrawave,
    ck_tile::GemmPipeline::COMPUTE_ASYNC_V2>;
// clang-format on

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
