// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Type combinations for BQuant tests - 1D GroupSize 128
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BQuant1D128Types = ::testing::Types<
    // 1d cases with grouping only on k axis
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8,  FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8,  PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8,  PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize1D_128>
>;
// clang-format on

// Test suite for BQuant 1D 128
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuant1D128Types);

// BQuant tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
