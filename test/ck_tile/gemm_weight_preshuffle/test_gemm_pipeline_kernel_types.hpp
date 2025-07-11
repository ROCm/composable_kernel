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

using WeightPreshuffle =
    ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::WeightPreshuffle>;

// clang-format off

using KernelTypesWeightPreshuffle = ::testing::Types<
     std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Default,        WeightPreshuffle>,
     std::tuple<    Row,     Col,     Row,       F8,         F8,         F32,       F16,             Default,        WeightPreshuffle>
>;

// clang-format on
