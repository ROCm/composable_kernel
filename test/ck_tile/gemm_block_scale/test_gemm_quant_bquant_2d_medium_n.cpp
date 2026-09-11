// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize2D32N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 32, 128>>;
using GroupSize2D64N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 64, 128>>;

// Type combinations for BQuant tests - 2D Medium N (32N and 64N)
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BQuant2DMediumNTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D32N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D32N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D32N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D32N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>
>;
// clang-format on

// Test suite for BQuant 2D Medium N
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuant2DMediumNTypes);

// BQuant tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
