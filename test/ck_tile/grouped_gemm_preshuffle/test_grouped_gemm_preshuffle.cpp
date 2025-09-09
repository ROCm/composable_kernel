// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_preshuffle_util.hpp"

using F16 = ck_tile::half_t;
using F8  = ck_tile::fp8_t;
using F32 = float;
using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,          F32,       F16>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGroupedGemmPreshuffle, KernelTypes);

#include "test_grouped_gemm_preshuffle_ut_cases.inc"
