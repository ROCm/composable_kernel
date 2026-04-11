// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using AQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>;

// Type combinations for AQuant tests - Mem Decode Intrawave Configuration
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using AQuantMemDecodeIntrawaveTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigDecodeIntrawave, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigDecodeIntrawave, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigDecodeIntrawave, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigDecodeIntrawave, GroupSize1D_128>
>;
// clang-format on

// Test suite for AQuant Mem Decode Intrawave
TYPED_TEST_SUITE(TestCkTileGemmAQuantMem, AQuantMemDecodeIntrawaveTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmAQuantMem, AQuantMemDecodeIntrawaveTest)
{
    this->run_test_with_validation(16, 64, 512);
}
