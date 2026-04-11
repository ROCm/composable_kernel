// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Type combinations for RowColQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using RowColQuantTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigBase, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, RowColQuant, GemmConfigBase, GroupSize1D_128>
>;
// clang-format on

// Test suite for RowColQuant
TYPED_TEST_SUITE(TestCkTileGemmRowColQuant, RowColQuantTypes);

// RowColQuant tests
TYPED_TEST(TestCkTileGemmRowColQuant, RowColQuantTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
