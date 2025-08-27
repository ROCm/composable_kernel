// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_util.hpp"

using F16  = ck_tile::half_t;
using F32  = float;
using F8   = ck_tile::fp8_t;
using BF16 = ck_tile::bf16_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using Default = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                           ck_tile::GemmPipelineScheduler::Default>;

using WeightPreshuffle =
    ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::WeightPreshuffle>;

// Adding alias for the F8 parameters to facilitate skipping tests.
// This alias can be removed once test failures are fixed.
using F8Types = std::tuple<Row, Col, Row, F8, F8, F32, F16, Default, WeightPreshuffle>;

// clang-format off

using KernelTypesWeightPreshuffle = ::testing::Types<
     std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,             Default,        WeightPreshuffle>,
     std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,      BF16,             Default,        WeightPreshuffle>,
     F8Types
     >;

// clang-format on
