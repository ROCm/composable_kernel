// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_util.hpp"

using I8  = ck_tile::int8_t;
using I32 = ck_tile::int32_t;

using F16 = ck_tile::half_t;
using F32 = float;
using F8  = ck_tile::fp8_t;

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

// clang-format off
using KernelTypesMem = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,       F8,         F32,       F16,             Interwave,         Mem>,
    std::tuple<    Row,     Row,     Row,       F8,       F8,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,       F8,         F32,       F16,             Interwave,         Mem>,
    std::tuple<    Row,     Col,     Row,       F8,       F8,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,       F8,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Col,     Row,     Row,       F8,       F8,         F32,       F16,             Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Interwave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,       F8,         F32,       F16,             Intrawave,         Mem>,
    std::tuple<    Col,     Col,     Row,       F8,       F8,         F32,       F16,             Interwave,         Mem>
>;

using KernelTypesCompV3 = ::testing::Types<
     std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV3>,
     std::tuple<    Row,     Row,     Row,       F8,       F8,         F32,       F16,             Intrawave,         CompV3>,
     std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV3>,
     std::tuple<    Row,     Col,     Row,       F8,       F8,         F32,       F16,             Intrawave,         CompV3>,
     std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV3>,
     std::tuple<    Col,     Row,     Row,       F8,       F8,         F32,       F16,             Intrawave,         CompV3>,
     std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV3>,
     std::tuple<    Col,     Col,     Row,       F8,       F8,         F32,       F16,             Intrawave,        CompV3>,
     std::tuple<    Row,     Row,     Row,       I8,       I8,         I32,       I32,             Intrawave,        CompV3>, 
     std::tuple<    Row,     Col,     Row,       I8,       I8,         I32,       I32,             Intrawave,        CompV3>,
     std::tuple<    Col,     Row,     Row,       I8,       I8,         I32,       I32,             Intrawave,        CompV3>, 
     std::tuple<    Col,     Col,     Row,       I8,       I8,         I32,       I32,             Intrawave,        CompV3>
    
>;

using KernelTypesCompV4 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV4>
>;

using KernelTypesPersistent = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV3,    Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Intrawave,        CompV3, NonPersistent>
>;

// clang-format on
