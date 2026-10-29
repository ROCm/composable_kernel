// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_mx_gemm_pipeline_util.hpp"
#include "test_mx_gemm_pipeline_prec_types.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using CompTDMV1 = ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::CompTDMV1>;
using CompTDMV2 = ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::CompTDMV2>;
using CompAsync = ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::CompAsync>;
using CompEightWaves =
    ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::CompEightWaves>;
using WeightPreshuffle =
    ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::WeightPreshuffle>;

using I16  = ck_tile::number<16>;
using I32  = ck_tile::number<32>;
using I64  = ck_tile::number<64>;
using I128 = ck_tile::number<128>;
using I256 = ck_tile::number<256>;
using I384 = ck_tile::number<384>;
using I512 = ck_tile::number<512>;

using ClusterEnable  = std::true_type;
using ClusterDisable = std::false_type;

// Optional tuple slot 18: warps along M and N (slots 16 and 17 must then be given too).
using Warps2x2 = ck_tile::sequence<2, 2>;
using Warps4x1 = ck_tile::sequence<4, 1>;
using Warps1x4 = ck_tile::sequence<1, 4>;
using Warps2x4 = ck_tile::sequence<2, 4>;

// clang-format off
// MX GEMM kernel types using TDM pipeline with scale support
// Tuple format:
//         ALayout, BLayout, CLayout, ADataType, BDataType, AScaleDataType, BScaleDataType, AccDataType, CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, N_TileSize, Scheduler, PipelineType, ScaleBlockSize
using KernelTypesMxGemmCompTDMWmma = ::testing::Types<
    // --- Scale32 (WarpTile=32, CompTDMV1) ---
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E5M3,  E5M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E4M3,  E4M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F4,    E8M0,  E5M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F8,    E5M3,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       BF8,       F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Row,     Row,       BF8,       F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    // --- Scale32 (WarpTile=32, CompTDMV2) ---
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E4M3,  E4M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F4,    E8M0,  E5M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F8,    E4M3,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       BF8,       F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32>,
    std::tuple<    Row,     Row,     Row,       BF8,       F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32>,
    std::tuple<    Row,     Row,     Row,       F4,        F4,    E5M3,  E5M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Col,     Row,     Row,       F4,        F8,    E5M3,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    // --- Scale16 (WarpTile=16, CompTDMV1) ---
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I16>,
    std::tuple<    Row,     Col,     Row,      BF8,       BF8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I16>,
    std::tuple<    Row,     Col,     Row,       F8,       BF8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I16>,
    std::tuple<    Row,     Col,     Row,      BF8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I16>,
    std::tuple<    Row,     Row,     Row,      BF8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I16>,  // RRR (non-RCR) layout
    // --- Scale16 (WarpTile=32, CompTDMV1) ---
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I16>,
    std::tuple<    Row,     Col,     Row,      BF8,       BF8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I16>,
    std::tuple<    Row,     Col,     Row,       F8,       BF8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I16>,
    std::tuple<    Row,     Col,     Row,      BF8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I16>,
    std::tuple<    Row,     Row,     Row,      BF8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I16>,  // RRR (non-RCR) layout
    // --- Scale16 (WarpTile=16, CompTDMV2) ---
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I16>,
    std::tuple<    Row,     Col,     Row,      BF8,       BF8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I16>,
    std::tuple<    Row,     Col,     Row,       F8,       BF8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I16>,
    std::tuple<    Row,     Col,     Row,      BF8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I16>,
    std::tuple<    Row,     Row,     Row,      BF8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I16>,  // RRR (non-RCR) layout
    // --- Scale16 (WarpTile=32, CompTDMV2) ---
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I16>,
    std::tuple<    Row,     Col,     Row,      BF8,       BF8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I16>,
    std::tuple<    Row,     Col,     Row,       F8,       BF8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I16>,
    std::tuple<    Row,     Col,     Row,      BF8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I16>,
    std::tuple<    Row,     Row,     Row,      BF8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I16>,  // RRR (non-RCR) layout
    // --- Scale32 cluster launch (from develop; ScaleBlockSize=I32 at idx 16, ClusterEnable at idx 17) ---
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32, std::false_type, ClusterEnable>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32, std::false_type, ClusterEnable>
>;

// bf16 C on the TDM pipelines, one row per data type and pipeline. All rows use the non-cluster
// 16x16 scale32 E8M0 path, which also runs on asic revision 0. Paired with the cases in
// test_mx_gemm_pipeline_smoke_cases.inc for the gfx1250 emulator smoke stage.
using KernelTypesMxGemmTdmBf16C = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I32>
>;

// FP6 (E2M3) on the TDM pipelines, one row per C type and pipeline. All rows use the one
// configuration the pipelines support for FP6: row-major A, column-major B, a 16x16x128 warp tile
// and 32-element E8M0 scales. Paired with the cases in test_mx_gemm_pipeline_smoke_cases.inc.
using KernelTypesMxGemmTdmFp6 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I32>
>;

// FP6 at K tiles of 256 and 384, for a few cases; the smoke cases run at a K tile of 128.
using KernelTypesMxGemmTdmFp6DeepKTile = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I256,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I384,       I16,        I16,        CompTDMV2,  I32>
>;

// FP6 with bf16 C at other block tiles and warp layouts: the smallest and a large block tile with
// 2 x 2 warps, four waves as 4 x 1 and 1 x 4, and eight waves (2 x 4) on V1.
using KernelTypesMxGemmTdmFp6TileLayouts = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,        I32,         I32,          I128,       I16,        I16,        CompTDMV2,  I32, std::false_type, ClusterDisable, Warps2x2>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,       I256,        I128,          I128,       I16,        I16,        CompTDMV1,  I32, std::false_type, ClusterDisable, Warps2x2>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,       I128,        I128,          I128,       I16,        I16,        CompTDMV1,  I32, std::false_type, ClusterDisable, Warps4x1>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I32, std::false_type, ClusterDisable, Warps1x4>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,       I128,        I128,          I128,       I16,        I16,        CompTDMV1,  I32, std::false_type, ClusterDisable, Warps2x4>
>;

// FP6 with bf16 C at a 128-deep K tile, one row per pipeline, for the K-loop cases of
// test_mx_gemm_tdm_fp6_kloop_shapes.cpp.
using KernelTypesMxGemmTdmFp6KLoop = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I32>
>;

using KernelTypesMxGemmCompAsyncRCR = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I256,       I16,        I16, CompAsync, I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I256,       I16,        I16, CompAsync, I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       I128,        I256,          I128,       I16,        I16, CompEightWaves, I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       I128,        I256,          I128,       I16,        I16, CompEightWaves, I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       I128,        I256,          I256,       I16,        I16, WeightPreshuffle, I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       I128,        I512,          I256,       I16,        I16, WeightPreshuffle, I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       I128,        I256,          I256,       I16,        I16, WeightPreshuffle, I32, std::true_type>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       I128,        I256,          I256,       I16,        I16, WeightPreshuffle, I32, std::true_type>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Row,     Col,     Row,      BF8,       BF8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Row,     Col,     Row,       F8,       BF8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>
>;

using KernelTypesMxGemmCompAsyncRCRLargeCases = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I256,       I16,        I16, CompAsync, I32>
>;

using KernelTypesMxGemmCompAsyncRRR = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Row,     Row,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Row,     Row,     Row,      BF8,       BF8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Row,     Row,     Row,       F8,       BF8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>
>;

using KernelTypesMxGemmCompAsyncCRR = ::testing::Types<
    std::tuple<    Col,     Row,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Col,     Row,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Col,     Row,     Row,      BF8,       BF8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Col,     Row,     Row,       F8,       BF8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>
>;

using KernelTypesMxGemmCompAsyncCCR = ::testing::Types<
    std::tuple<    Col,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Col,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Col,     Col,     Row,      BF8,       BF8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>,
    std::tuple<    Col,     Col,     Row,       F8,       BF8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I256,       I32,        I32, CompAsync, I32>
>;

// clang-format on
