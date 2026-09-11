// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using AQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>;

// Type combinations for AQuant tests - PreshuffleQuant Configurations
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using AQuantPreshuffleTypes = ::testing::Types<
    // PreshuffleQuant = true && TransposeC = false (with RowMajor AQ - PreshuffleQuant only supports RowMajor)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize1D_128>,

    // PreshuffleQuant = true && TransposeC = true (with RowMajor AQ - PreshuffleQuant only supports RowMajor)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize1D_128>
>;
// clang-format on

// Test suite for AQuant Preshuffle
TYPED_TEST_SUITE(TestCkTileGemmAQuant, AQuantPreshuffleTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmAQuant, AQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
