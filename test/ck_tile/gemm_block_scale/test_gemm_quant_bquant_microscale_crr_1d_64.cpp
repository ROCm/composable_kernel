// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize64 = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;

// Type combinations for BQuant tests - 1D GroupSize 64
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BQuant1D64Types = ::testing::Types<
    // CRR BQ: C
    std::tuple<ColumnMajor, RowMajor, RowMajor, ColumnMajor, BF16,   BF8,  E8M0, BF16, BQuantGrouped,    GemmConfigMx, GroupSize64>,
    // CRR BQ: R
    std::tuple<ColumnMajor, RowMajor, RowMajor,    RowMajor, BF16,  BF16,  E8M0, BF16, BQuantGrouped,    GemmConfigMx, GroupSize64>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, ColumnMajor, BF16, PkFP4,  E8M0, BF16, BQuantGrouped, GemmConfigMxFP4, GroupSize64>
>;
// clang-format on

// Test suite for BQuant 1D 64
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuant1D64Types);

// BQuant tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
