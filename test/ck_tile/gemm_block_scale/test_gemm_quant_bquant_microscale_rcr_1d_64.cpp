// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize64 = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;

// Type combinations for BQuant tests - 1D GroupSize 64
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BQuant1D64Types = ::testing::Types<
    // RCR BQ: C
    std::tuple< RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF16, PkFP4, E8M0, BF16, BQuantGrouped, GemmConfigMx, GroupSize64>,
    std::tuple< RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF16,   BF8, E8M0, BF16, BQuantGrouped, GemmConfigMx, GroupSize64>,
    std::tuple< RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF16,  BF16, E8M0, BF16, BQuantGrouped, GemmConfigMx, GroupSize64>,
    std::tuple< RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP16,  FP16, E8M0, FP16, BQuantGrouped, GemmConfigMx, GroupSize64>,
    std::tuple< RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF16,   FP8, E8M0, BF16, BQuantGrouped, GemmConfigMx, GroupSize64>,
    std::tuple< RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP16,   FP8, E8M0, FP16, BQuantGrouped, GemmConfigMx, GroupSize64>,
    std::tuple< RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP16,   BF8, E8M0, FP16, BQuantGrouped, GemmConfigMx, GroupSize64>,
    std::tuple< RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP16, PkFP4, E8M0, FP16, BQuantGrouped, GemmConfigMx, GroupSize64>,
    // RCR BQ: R
    std::tuple< RowMajor, ColumnMajor, RowMajor,    RowMajor, BF16,   BF8, E8M0, BF16, BQuantGrouped, GemmConfigMx, GroupSize64>,
    std::tuple< RowMajor, ColumnMajor, RowMajor,    RowMajor, BF16,  BF16, E8M0, BF16, BQuantGrouped, GemmConfigMx, GroupSize64>
>;
// clang-format on

// Test suite for BQuant 1D 64
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuant1D64Types);

// BQuant tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
