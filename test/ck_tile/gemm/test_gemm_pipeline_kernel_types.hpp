// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_prec_types.hpp"
#include "test_gemm_pipeline_util.hpp"

#include "ck_tile/host.hpp"

#include "gtest/gtest.h"

#include <tuple>
#include <type_traits>

using Row       = ck_tile::tensor_layout::gemm::RowMajor;
using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
using Intrawave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Intrawave>;
using Interwave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Interwave>;

using Mem       = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Mem>;
using CompV3    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV3>;
using CompV4    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV4>;
using CompV6    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV6>;
using CompAsync = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompAsync>;
using CompAsyncEightWaves =
    ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompAsyncEightWaves>;
using CompTDMV1 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompTDMV1>;
using CompTDMV2 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompTDMV2>;

using Persistent    = std::true_type;
using NonPersistent = std::false_type;

using ClusterEnable  = std::true_type;
using ClusterDisable = std::false_type;

using I8   = ck_tile::number<8>;
using I16  = ck_tile::number<16>;
using I32  = ck_tile::number<32>;
using I64  = ck_tile::number<64>;
using I128 = ck_tile::number<128>;
using I192 = ck_tile::number<192>;
using I256 = ck_tile::number<256>;

// clang-format off
using KernelTypesMem = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, N_TileSize, Scheduler, PipelineType
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Interwave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Interwave,         Mem>
>;

using KernelTypesMemWmma = ::testing::Types<
#ifdef CK_USE_GFX1250
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I32,         I32,          I64,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I32,         I32,          I64,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F16,       F16,        I32,         I32,          I64,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F16,       F16,        I32,         I32,          I64,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F16,       F16,        I32,         I32,          I64,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F16,       F16,        I32,         I32,          I64,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I32,         I32,          I64,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I32,         I32,          I64,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I32,         I32,          I64,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I32,         I32,          I64,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I32,         I32,          I64,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I32,         I32,          I64,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I32,         I32,          I64,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I32,         I32,          I64,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I32,         I32,          I64,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I32,         I32,          I64,        I16,        I16, Interwave,         Mem>,
#else
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16, Intrawave,         Mem>,
#ifdef CK_USE_WMMA_FP8
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
#endif
#endif
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,       I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,       I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Interwave,         Mem>
>;

using KernelTypesCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3>
>;

#ifdef CK_USE_GFX1250
#define MinK  I64
#else
#define MinK I32
#endif
using KernelTypesCompV3Wmma = ::testing::Types<
#ifdef CK_USE_GFX1250
    std::tuple<    Row,     Col,     Row,       F4,        F4,          F32,       F16,        I128,        I128,         I128,       I32,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,          F32,       F16,        I128,        I128,         I128,       I32,        I32, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F32,       F32,         F32,       F32,        I64,         I64,          I8,         I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F32,       F32,         F32,       F32,        I64,         I64,          I8,         I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F32,       F32,         F32,       F32,        I64,         I64,          I8,         I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F32,       F32,         F32,       F32,        I64,         I64,          I8,         I16,        I16, Intrawave,        CompV3>,
#endif
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,   
    std::tuple<    Col,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
#ifdef CK_USE_WMMA_FP8
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        BF8,         F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,        I4,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF8,       I4,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>, 
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        BF8,         F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,        I4,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF8,       I4,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        BF8,         F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,        I4,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF8,       I4,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        BF8,         F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,        I4,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF8,       I4,          F32,       F16,        I64,         I64,          MinK,       I16,        I16, Intrawave,        CompV3>,
#endif
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F16,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       BF16,      I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       BF16,      I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       BF16,      I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       BF16,      I4,          F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3>
>;

using KernelTypesCompV4 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       BF16,      I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F8,        BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F8,        I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       BF8,       I4,          F32,       F16,        I256,        I256,         I32,        I32,        I32, Intrawave,        CompV4>
>;


using KernelTypesCompTDMWmma = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompTDMV1>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompTDMV1, NonPersistent, ClusterEnable>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompTDMV2>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompTDMV2, NonPersistent, ClusterEnable>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompTDMV1>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompTDMV1, NonPersistent, ClusterEnable>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompTDMV1>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompTDMV2>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompTDMV2, NonPersistent, ClusterEnable>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompTDMV2>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,          F32,       F16,        I64,         I64,          I128,       I16,        I16, Intrawave,        CompTDMV1>,
    std::tuple<    Row,     Col,     Row,       F8,        F4,          F32,       F16,        I64,         I64,          I128,       I16,        I16, Intrawave,        CompTDMV1>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,          F32,       F16,        I64,         I64,          I128,       I32,        I32, Intrawave,        CompTDMV1>,
    std::tuple<    Row,     Col,     Row,       F8,        F4,          F32,       F16,        I64,         I64,          I128,       I32,        I32, Intrawave,        CompTDMV1>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I128,       I16,        I16, Intrawave,        CompTDMV1>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I128,       I16,        I16, Intrawave,        CompTDMV1>
>;

using KernelTypesCompAsyncWmma = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompAsync>
>;

// clang-format on
template <typename ALayout, typename BLayout, typename CLayout, typename InputType>
using CompAsyncConfig = std::tuple<ALayout,
                                   BLayout,
                                   CLayout,
                                   InputType, // AType
                                   InputType, // BType
                                   F32,       // AccType
                                   F16,       // OutputType
                                   I256,      // MBlockTileSize
                                   I256,      // NBlockTileSize
                                   I64,       // KBlockTileSize
                                   I32,       // MWarpTileSize
                                   I32,       // NWarpTileSize
                                   Intrawave,
                                   CompAsync>;

template <typename ALayout, typename BLayout, typename CLayout, typename InputType>
using CompAsyncConfig16x16x128 = std::tuple<ALayout,
                                            BLayout,
                                            CLayout,
                                            InputType, // AType
                                            InputType, // BType
                                            F32,       // AccType
                                            F16,       // OutputType
                                            I64,       // MBlockTileSize
                                            I64,       // NBlockTileSize
                                            I128,      // KBlockTileSize
                                            I16,       // MWarpTileSize
                                            I16,       // NWarpTileSize
                                            Intrawave,
                                            CompAsync>;

template <typename ALayout, typename BLayout, typename CLayout, typename InputType>
using CompAsyncEightWavesConfig4Bit = std::tuple<ALayout,
                                                 BLayout,
                                                 CLayout,
                                                 InputType, // AType
                                                 InputType, // BType
                                                 F32,       // AccType
                                                 F16,       // OutputType
                                                 I128,      // MBlockTileSize
                                                 I256,      // NBlockTileSize
                                                 I256,      // KBlockTileSize
                                                 I16,       // MWarpTileSize
                                                 I16,       // NWarpTileSize
                                                 Intrawave,
                                                 CompAsyncEightWaves>;

template <typename ALayout, typename BLayout, typename CLayout, typename InputType>
using CompAsyncEightWavesConfig8BitFP = std::tuple<ALayout,
                                                   BLayout,
                                                   CLayout,
                                                   InputType, // AType
                                                   InputType, // BType
                                                   F32,       // AccType
                                                   F16,       // OutputType
                                                   I128,      // MBlockTileSize
                                                   I256,      // NBlockTileSize
                                                   I128,      // KBlockTileSize
                                                   I16,       // MWarpTileSize
                                                   I16,       // NWarpTileSize
                                                   Intrawave,
                                                   CompAsyncEightWaves>;

template <typename ALayout, typename BLayout, typename CLayout, typename InputType>
using CompAsyncEightWavesConfig8BitINT = std::tuple<ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    InputType, // AType
                                                    InputType, // BType
                                                    INT32,     // AccType
                                                    INT32,     // OutputType
                                                    I128,      // MBlockTileSize
                                                    I256,      // NBlockTileSize
                                                    I128,      // KBlockTileSize
                                                    I16,       // MWarpTileSize
                                                    I16,       // NWarpTileSize
                                                    Intrawave,
                                                    CompAsyncEightWaves>;

template <typename ALayout, typename BLayout, typename CLayout, typename InputType>
using CompAsyncEightWavesConfig16Bit = std::tuple<ALayout,
                                                  BLayout,
                                                  CLayout,
                                                  InputType, // AType
                                                  InputType, // BType
                                                  F32,       // AccType
                                                  F16,       // OutputType
                                                  I192,      // MBlockTileSize
                                                  I256,      // NBlockTileSize
                                                  I64,       // KBlockTileSize
                                                  I16,       // MWarpTileSize
                                                  I16,       // NWarpTileSize
                                                  Intrawave,
                                                  CompAsyncEightWaves>;

using KernelTypesCompAsync = ::testing::Types<CompAsyncConfig<Row, Row, Row, F16>,
                                              CompAsyncConfig<Row, Col, Row, F16>,
                                              CompAsyncConfig<Col, Row, Row, F16>,
                                              CompAsyncConfig<Col, Col, Row, F16>,
                                              CompAsyncConfig<Row, Row, Row, F8>,
                                              CompAsyncConfig<Row, Col, Row, F8>,
                                              CompAsyncConfig<Col, Row, Row, F8>,
                                              CompAsyncConfig<Col, Col, Row, F8>>;

using KernelTypesCompAsync16x16x128 = ::testing::Types<CompAsyncConfig16x16x128<Row, Col, Row, F4>,
                                                       CompAsyncConfig16x16x128<Row, Col, Row, F8>>;

using KernelTypesCompAsyncEightWaves =
    ::testing::Types<CompAsyncEightWavesConfig8BitINT<Row, Col, Row, INT8>,
                     CompAsyncEightWavesConfig8BitFP<Row, Col, Row, F8>,
                     CompAsyncEightWavesConfig8BitFP<Row, Col, Row, BF8>,
                     CompAsyncEightWavesConfig4Bit<Row, Col, Row, F4>,
                     CompAsyncEightWavesConfig16Bit<Row, Col, Row, F16>,
                     CompAsyncEightWavesConfig16Bit<Row, Col, Row, BF16>>;

// clang-format off
using KernelTypesCompV6 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Col,     Row,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Col,     Col,     Row,       BF8,       BF8,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Row,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Col,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32, Intrawave,        CompV6>
>;

using KernelTypesCompV4Wmma = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV4>
>;


using KernelTypesPersistent = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, M_TileSize, K_TileSize, Scheduler,  PipelineType,    Persistent
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32, Intrawave,        CompV3, NonPersistent>
>;

using KernelTypesPersistentWmma = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16, Intrawave,        CompV3, NonPersistent>
>;

// TF32 (gfx950 only): 3x bf16 MFMA emulation
// Tile: 128x128x64, Warp tile: 32x32x16
using KernelTypesTf32Mem = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, N_TileSize, K_TileSize, Scheduler, PipelineType
    std::tuple<    Row,     Row,     Row,      TF32,      TF32,         F32,       F32,        I128,        I128,         I64,        I32,        I32, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,      TF32,      TF32,         F32,       F32,        I128,        I128,         I64,        I32,        I32, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,      TF32,      TF32,         F32,       F32,        I128,        I128,         I64,        I32,        I32, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,      TF32,      TF32,         F32,       F32,        I128,        I128,         I64,        I32,        I32, Interwave,         Mem>
>;

// clang-format on
