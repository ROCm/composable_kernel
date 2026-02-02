// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <gtest/gtest.h>
#include <memory>

#include "test_gemm_quant_fixtures.hpp"

// Type aliases for readability
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using Half        = ck_tile::half_t;
using PkFP4       = ck_tile::pk_fp4_t;
using ABQuantGrouped =
    std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::ABQuantGrouped>;

// 1d block sizes for AQuant
using GroupSize1D = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

// 2d block sizes for BQuant
using GroupSize2D = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// Type combinations for ABQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantTypes = ::testing::Types<
    // PreshuffleQuant = false && TransposeC = false
    // RCR layout with RowMajor AQ, ColumnMajor BQ
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkFP4, PkFP4, float, Half, ABQuantGrouped, GemmConfigPadding, GroupSize1D, GroupSize2D, ColumnMajor>
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
