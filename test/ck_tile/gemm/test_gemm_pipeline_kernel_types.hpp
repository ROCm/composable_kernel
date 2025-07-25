// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

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

// clang-format off
using KernelTypesMem = ::testing::Types<
//               ALayout   BLayout  CLayout  ADataType   BDataType  AccDataType  CDataType          Scheduler       PipelineType        SkipALds               SkipBLds
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,             Interwave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Row,     Row,     Row,       F8,        F8,          F32,       F16,             Intrawave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,             Interwave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16,             Intrawave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,             Intrawave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,          F32,       F16,             Interwave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,             Intrawave,         Mem,          std::false_type,      std::false_type>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,          F32,       F16,             Interwave,         Mem,          std::false_type,      std::false_type>,
    // SkipALds/SkipBLds
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem,          std::false_type,      std::true_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem,          std::true_type,       std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem,          std::true_type,       std::true_type>
>;

using KernelTypesCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,      F16,         F32,       F16,             Intrawave,         CompV3,        std::false_type,      std::false_type>,
    std::tuple<    Row,     Row,     Row,       F8,       F8,          F32,       F16,             Intrawave,         CompV3,        std::false_type,      std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,      F16,         F32,       F16,             Intrawave,         CompV3,        std::false_type,      std::false_type>,
    std::tuple<    Row,     Col,     Row,       F8,       F8,          F32,       F16,             Intrawave,         CompV3,        std::false_type,      std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,      F16,         F32,       F16,             Intrawave,         CompV3,        std::false_type,      std::false_type>,
    std::tuple<    Col,     Row,     Row,       F8,       F8,          F32,       F16,             Intrawave,         CompV3,        std::false_type,      std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,      F16,         F32,       F16,             Intrawave,         CompV3,        std::false_type,      std::false_type>,
    std::tuple<    Col,     Col,     Row,       F8,       F8,          F32,       F16,             Intrawave,         CompV3,        std::false_type,      std::false_type>
>;

using KernelTypesCompV4 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4,        std::false_type,      std::false_type>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4,        std::false_type,      std::false_type>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4,        std::false_type,      std::false_type>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4,        std::false_type,      std::false_type>
>;

// clang-format on
