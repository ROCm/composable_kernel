// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Type combinations for ABQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantTypes = ::testing::Types<
    // RCR layout with RowMajor AQ, ColumnMajor BQ
    // PreshuffleB = true && TransposeC = false
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkFP4, PkFP4, float, Half, ABQuantGrouped, GemmConfigPreshuffleBPrefill, GroupSize1D_128, GroupSize2D, ColumnMajor>
>;
// clang-format on

// Test suite for ABQuant
TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
