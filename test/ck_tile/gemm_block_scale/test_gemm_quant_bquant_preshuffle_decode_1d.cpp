// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Type combinations for BQuant Preshuffle tests - Decode Config 1D
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BPreshuffleDecode1DTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize1D_128>
>;
// clang-format on

// Test suite for BQuant Preshuffle Decode 1D
TYPED_TEST_SUITE(TestCkTileGemmPreshuffleBBQuant, BPreshuffleDecode1DTypes);

// BQuant PreshuffleB tests
TYPED_TEST(TestCkTileGemmPreshuffleBBQuant, BQuantPreshuffleTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
