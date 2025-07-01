// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_util.hpp"

using F16       = ck_tile::half_t;
using F32       = float;
using F8        = ck_tile::fp8_t;
using Row       = ck_tile::tensor_layout::gemm::RowMajor;
using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
using Intrawave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Intrawave>;
using Interwave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Interwave>;
using Mem       = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Mem>;
using CompV3    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV3>;
using CompV4    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV4>;

using Persistent    = std::true_type;
using NonPersistent = std::false_type;

// clang-format off
using KernelTypesMem = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType,              M_N_BlockSize,              K_BlockSize,              M_N_TileSize,                K_TileSize, Scheduler, PipelineType
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Interwave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Interwave,         Mem>
>;

// TODO: F8 not ready on both GFX11/GFX12 yet. Will uncomment those F8 case when F8 implementation is ready
using KernelTypesMemWmma = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Interwave,         Mem>,
    //std::tuple<    Row,     Row,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Interwave,         Mem>,
    //std::tuple<    Row,     Row,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Interwave,         Mem>,
    //std::tuple<    Row,     Col,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Interwave,         Mem>,
    //std::tuple<    Row,     Col,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Interwave,         Mem>,
    //std::tuple<    Col,     Row,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,         Mem>,
    //std::tuple<    Col,     Row,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Interwave,         Mem>
    //std::tuple<    Col,     Col,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,         Mem>,
    //std::tuple<    Col,     Col,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Interwave,         Mem>
>;

using KernelTypesCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV3>
>;

using KernelTypesCompV3Wmma = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV3>,
    // std::tuple<    Row,     Row,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,         CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV3>,
    // std::tuple<    Row,     Col,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,         CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV3>,
    // std::tuple<    Col,     Row,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,         CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,      F16,        F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV3>
    // std::tuple<    Col,     Col,     Row,       F8,       F8,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV3>
>;

using KernelTypesCompV4 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<32>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<32>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<32>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<32>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV4>
>;

using KernelTypesCompV4Wmma = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV4>
>;


using KernelTypesPersistent = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType,              M_N_BlockSize,             K_BlockSize,               M_N_TileSize,                K_TileSize, Scheduler,  PipelineType,    Persistent
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<256>,      ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>, Intrawave,        CompV3, NonPersistent>
>;

using KernelTypesPersistentWmma = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,       ck_tile::number<64>,       ck_tile::number<32>,       ck_tile::number<16>,       ck_tile::number<16>, Intrawave,        CompV3, NonPersistent>
>;

// clang-format on
