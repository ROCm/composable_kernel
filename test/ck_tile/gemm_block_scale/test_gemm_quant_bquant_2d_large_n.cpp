// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// Type combinations for BQuant tests - 2D Large N (128N)
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BQuant2DLargeNTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D128N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D128N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D128N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D128N>
>;
// clang-format on

// Test suite for BQuant 2D Large N
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuant2DLargeNTypes);

// BQuant tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
