// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize64    = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;
using GroupSize2D64N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 64, 128>>;

// Type combinations for BQuant tests - Transpose Layouts
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BQuantTransposeTypes = ::testing::Types<
    // some cases with transpose layouts
    std::tuple<   RowMajor,    RowMajor, RowMajor, RowMajor,    FP8, FP8, float, Half, BQuantGrouped, GemmConfigBase, GroupSize64>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, BQuantGrouped, GemmConfigBase, GroupSize64>,
    std::tuple<ColumnMajor,    RowMajor, RowMajor, RowMajor,    FP8, FP8, float, Half, BQuantGrouped, GemmConfigBase, GroupSize64>,
    std::tuple<   RowMajor,    RowMajor, RowMajor, RowMajor,    FP8, FP8, float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<ColumnMajor,    RowMajor, RowMajor, RowMajor,    FP8, FP8, float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,

    // pkint4 + transpose cases
    std::tuple<   RowMajor,    RowMajor, RowMajor,    RowMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigBase, GroupSize64>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigBase, GroupSize64>,
    std::tuple<ColumnMajor,    RowMajor, RowMajor,    RowMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigBase, GroupSize64>,
    std::tuple<   RowMajor,    RowMajor, RowMajor,    RowMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<ColumnMajor,    RowMajor, RowMajor,    RowMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>
>;
// clang-format on

// Test suite for BQuant Transpose
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuantTransposeTypes);

// BQuant tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
