// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_util.hpp"

using INT8  = ck_tile::int8_t;
using INT32 = ck_tile::int32_t;

using F16 = ck_tile::half_t;
using F32 = float;
using F8  = ck_tile::fp8_t;

using BF16 = ck_tile::bf16_t;
using BF8  = ck_tile::bf8_t;

using Row       = ck_tile::tensor_layout::gemm::RowMajor;
using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
using Intrawave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Intrawave>;
using Interwave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Interwave>;

using Mem    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Mem>;
using CompV3 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV3>;
using CompV4 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV4>;

using Persistent    = std::true_type;
using NonPersistent = std::false_type;

using I16  = ck_tile::number<16>;
using I32  = ck_tile::number<32>;
using I64  = ck_tile::number<64>;
using I256 = ck_tile::number<256>;

// clang-format off
using KernelTypesMem = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, M_TileSize, K_TileSize, Scheduler, PipelineType, SkipALds.      , SkipBLds
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    // SkipALds/SkipBLds
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::false_type, std::true_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::true_type,  std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Interwave,         Mem,  std::true_type,  std::true_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::false_type, std::true_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::true_type,  std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,         Mem,  std::true_type,  std::true_type>
>;

using KernelTypesMemWmma = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,       I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       BF16,       I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,         Mem,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Interwave,         Mem,  std::false_type, std::false_type>
>;

using KernelTypesCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>
>;

using KernelTypesCompV3Wmma = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       BF16,      BF16,        F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       INT8,      INT8,        INT32,     INT32,      I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       BF8,       BF8,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,  std::false_type, std::false_type>
>;

using KernelTypesCompV4 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I32,        I32,        I32,        I16, Intrawave,        CompV4,  std::false_type, std::false_type>
>;

using KernelTypesCompV4Wmma = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV4,  std::false_type, std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV4,  std::false_type, std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV4,  std::false_type, std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV4,  std::false_type, std::false_type>
>;


using KernelTypesPersistent = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, M_TileSize, K_TileSize, Scheduler,  PipelineType,    Persistent
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I64,        I32,        I32,        I16, Intrawave,        CompV3, NonPersistent>
>;

using KernelTypesPersistentWmma = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I64,         I64,          I32,        I16,        I16,        I16, Intrawave,        CompV3, NonPersistent>
>;

// clang-format on
