// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_multiple_d_gemm_util.hpp"

using F16 = ck_tile::half_t;
using F32 = float;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypes = ::testing::Types<
    //          ALayout, BLayout, CLayout, D0Layout, D1Layout, ADataType, BDataType, D0DataType,  D0DataType, AccDataType, CDataType
    std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F16,        F16,        F32,      F16>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileMultipleDGemm, KernelTypes);

#include "test_multiple_d_gemm_ut_cases.inc"
