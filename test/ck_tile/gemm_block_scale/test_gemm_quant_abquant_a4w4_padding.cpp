// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Type combinations for ABQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantTypes = ::testing::Types<
    // PreshuffleQuant = false && TransposeC = false
    // RCR layout with RowMajor AQ, ColumnMajor BQ
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkFP4, PkFP4, float, Half, ABQuantGrouped, GemmConfigPadding, GroupSize1D_128, GroupSize2D, ColumnMajor>
>;
// clang-format on

// Test suite for ABQuant
TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantTypes);

// AQuant tests

TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest_PadK)
{
    this->run_test_with_validation(1024, 1024, 832);
}

TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest_PadN)
{
    this->run_test_with_validation(1024, 832, 1024);
}

TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest_PadM)
{
    this->run_test_with_validation(832, 1024, 1024);
}

TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest_PadMNK)
{
    this->run_test_with_validation(832, 832, 832);
}

TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest_PadNK)
{
    this->run_test_with_validation(1024, 832, 832);
}
