// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Type combinations for BQuant Preshuffle tests - Prefill Config 1D
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BPreshufflePrefill1DTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, BQuantGrouped, GemmConfigPreshuffleQuantPrefill, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8, float, Half, BQuantGrouped, GemmConfigPreshuffleQuantPrefill, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigPreshuffleQuantPrefill, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, PkInt4, BF8, Half, BQuantGrouped, GemmConfigPreshuffleQuantPrefill, GroupSize1D_128>
>;
// clang-format on

// Test suite for BQuant Preshuffle Prefill 1D
TYPED_TEST_SUITE(TestCkTileGemmPreshuffleBBQuant, BPreshufflePrefill1DTypes);

// BQuant PreshuffleB tests
TYPED_TEST(TestCkTileGemmPreshuffleBBQuant, BQuantPreshuffleTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
