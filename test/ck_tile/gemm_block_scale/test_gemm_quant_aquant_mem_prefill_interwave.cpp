// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using AQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>;

// Type combinations for AQuant tests - Mem Prefill Interwave Configuration
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using AQuantMemPrefillInterwaveTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigPrefillInterwave, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigPrefillInterwave, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigPrefillInterwave, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigPrefillInterwave, GroupSize1D_128>
>;
// clang-format on

// Test suite for AQuant Mem Prefill Interwave
TYPED_TEST_SUITE(TestCkTileGemmAQuantMem, AQuantMemPrefillInterwaveTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmAQuantMem, AQuantMemPrefillInterwaveTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
