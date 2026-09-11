// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// Type combinations for ABQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantTypes = ::testing::Types<
    // 1D BScales; PreshuffleQuant = false && TransposeC = false (RCR layout with RowMajor AQ)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigBase, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigBase, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigBase, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigBase, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigBase, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigBase, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
	
    // 2D B-scales; PreshuffleQuant = false && TransposeC = true (RCR layout with RowMajor AQ)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigTransposeC, GroupSize1D_128, GroupSize2D128N, ColumnMajor>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigTransposeC, GroupSize1D_128, GroupSize2D128N, ColumnMajor>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigTransposeC, GroupSize1D_128, GroupSize2D128N, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigTransposeC, GroupSize1D_128, GroupSize2D128N, ColumnMajor>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigTransposeC, GroupSize1D_128, GroupSize2D128N, ColumnMajor>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigTransposeC, GroupSize1D_128, GroupSize2D128N, ColumnMajor>
>;
// clang-format on

// Test suite for ABQuant
TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
