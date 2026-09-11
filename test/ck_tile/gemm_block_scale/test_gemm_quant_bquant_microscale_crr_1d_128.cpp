// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize128 = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

// Type combinations for BQuant tests - 1D GroupSize 128
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BQuant1D128Types = ::testing::Types<
    // CRR BQ: C
    std::tuple<ColumnMajor, RowMajor, RowMajor, ColumnMajor, BF16,   BF8,  E8M0, BF16, BQuantGrouped,    GemmConfigMx, GroupSize128>,
    // CRR BQ: R
    std::tuple<ColumnMajor, RowMajor, RowMajor,    RowMajor, BF16,  BF16,  E8M0, BF16, BQuantGrouped,    GemmConfigMx, GroupSize128>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, ColumnMajor, BF16, PkFP4,  E8M0, BF16, BQuantGrouped, GemmConfigMxFP4, GroupSize128>
>;
// clang-format on

// Test suite for BQuant 1D 128
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuant1D128Types);

// BQuant tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
