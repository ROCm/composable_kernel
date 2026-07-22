// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Type combinations for RowColQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using RowColQuantTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigBase, GroupSize1D_128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, RowColQuant, GemmConfigBase, GroupSize1D_128>
>;

using RowColQuantMultiDTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, float, RowColQuant, GemmConfigBase, GroupSize1D_128, GroupSize1D_128, RowMajor, ck_tile::tuple<float>, ck_tile::tuple<RowMajor>>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigBase, GroupSize1D_128, GroupSize1D_128, RowMajor, ck_tile::tuple<float, Half>, ck_tile::tuple<RowMajor, RowMajor>>
>;
// clang-format on

// Test suite for RowColQuant
TYPED_TEST_SUITE(TestCkTileGemmRowColQuant, RowColQuantTypes);

// RowColQuant tests
TYPED_TEST(TestCkTileGemmRowColQuant, RowColQuantTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

template <typename Tuple>
class TestCkTileGemmRowColQuantMultiD : public TestCkTileGemmRowColQuant<Tuple>
{
};

// Test suite for RowColQuant with multiple Ds
TYPED_TEST_SUITE(TestCkTileGemmRowColQuantMultiD, RowColQuantMultiDTypes);

// RowColQuant tests
TYPED_TEST(TestCkTileGemmRowColQuantMultiD, RowColQuantMultiDTest)
{
    this->run_test_with_validation(1024, 2048, 512);
}
