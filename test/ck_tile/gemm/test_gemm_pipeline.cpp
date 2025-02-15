// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_util.hpp"

using F16       = ck_tile::half_t;
using F32       = float;
using Row       = ck_tile::tensor_layout::gemm::RowMajor;
using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
using Intrawave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Intrawave>;
using Interwave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Interwave>;
using Mem       = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Mem>;
using CompV3    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV3>;
using CompV4    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV4>;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, GemmPipelineScheduler, PipelineType
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV3>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV3>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV3>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV3>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmPipeline, KernelTypes);

#include "test_gemm_pipeline_ut_cases.inc"
