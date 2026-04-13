// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using AQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>;

// Type combinations for AQuant tests - CCR layout
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using AQuantBaseCCRTypes = ::testing::Types<
    // CCR layout (ColumnMajor A, ColumnMajor B, RowMajor C with ColumnMajor AQ) - NEW layout support
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize1D_128>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize1D_128>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize1D_128>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigBase, GroupSize1D_128>
>;
// clang-format on

// Test suite for AQuant Base CCR
TYPED_TEST_SUITE(TestCkTileGemmAQuant, AQuantBaseCCRTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmAQuant, AQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
