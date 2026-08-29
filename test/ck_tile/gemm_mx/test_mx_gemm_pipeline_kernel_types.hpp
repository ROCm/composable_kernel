// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
using I512 = ck_tile::number<512>;

using ClusterEnable  = std::true_type;
using ClusterDisable = std::false_type;

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
