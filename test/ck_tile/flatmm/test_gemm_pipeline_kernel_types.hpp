// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_util.hpp"

using F16 = ck_tile::half_t;
using F32 = float;
using F8  = ck_tile::fp8_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using Default = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                           ck_tile::GemmPipelineScheduler::Default>;

using Flatmm = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Flatmm>;

// clang-format off

using KernelTypesFlatmm = ::testing::Types<
     std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Default,        Flatmm>,
     std::tuple<    Row,     Col,     Row,       F8,         F8,         F32,       F16,             Default,        Flatmm>
>;

// clang-format on
