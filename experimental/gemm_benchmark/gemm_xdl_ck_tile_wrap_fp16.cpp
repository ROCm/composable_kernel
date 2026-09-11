// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include "common.hpp"
// #define CK_TILE_WRAP_ENABLE_BPRESHUFFLE 1
#include "gemm_xdl_ck_tile_wrap.hpp"

#if 1
using ADataType       = ck::half_t;
using BDataType       = ck::half_t;
using ComputeDataType = ck::half_t;
#else
using ADataType       = ck::pk_i4_t;
using BDataType       = ck::pk_i4_t;
using ComputeDataType = ck::f8_t;
#endif
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
#ifdef CK_TILE_WRAP_ENABLE_BPRESHUFFLE
static constexpr ck::index_t M_Warp = 1;
static constexpr ck::index_t N_Warp = 4;
static constexpr auto PipelineVer   = ck_tile::GemmPipeline::PRESHUFFLE_FLATMM;
#else
static constexpr ck::index_t M_Warp = 2;
static constexpr ck::index_t N_Warp = 2;
static constexpr auto PipelineVer   = ck_tile::GemmPipeline::COMPUTE_V3;
#endif

// clang-format off
using DeviceGemmV2Instance = ck::tensor_operation::device::DeviceGemm_Xdl_CkTileWrap<
    ALayout, BLayout, CLayout,
    ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,
    AElementOp, BElementOp, CElementOp,
    GemmDefault,
    128, 128, 128,                                      // M/N/K PerBlock
    16, 16, ck_tile::get_k_warp_tile<decltype(ck::GetCkTileDataType<ComputeDataType>()), 16>(),  // M/N/K PerXDL
    M_Warp, N_Warp, 1,                                         // M/N/K Warp
    2,
    ComputeDataType,
    1, 1,
    ck_tile::GemmPipelineScheduler::Intrawave,
    PipelineVer>;
// clang-format on

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[])
{
    return !run_gemm_splitk_example<DeviceGemmV2Instance::GemmConfig::Preshuffle>(argc, argv);
}
