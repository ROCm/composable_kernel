// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize128 = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

// Type combinations for BQuant split-K tests - Prefill shape, GroupSize 128
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BQuantSplitKPrefillTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigPrefill, GroupSize128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigPrefill, GroupSize128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigPrefill, GroupSize128>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigPrefill, GroupSize128>
>;
// clang-format on

// Test suite for BQuant split-K Prefill
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuantSplitKPrefillTypes);

// BQuant split-K tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedSplitK2Test)
{
    // K=1024 for split_k=2: 1024/2=512=4×128 ✓
    // K must be divisible by K_Tile(128)*split_k(2)=256
    this->run_test_with_validation(128, 128, 1024, 2);
}

TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedSplitK3Test)
{
    // K=3072 for split_k=3: 3072/3=1024=8×128 ✓
    // K must be divisible by K_Tile(128)*split_k(3)=384
    this->run_test_with_validation(128, 128, 3072, 3);
}

TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedSplitK4Test)
{
    // K=2048 for split_k=4: 2048/4=512=4×128 ✓
    // K must be divisible by K_Tile(128)*split_k(4)=512
    this->run_test_with_validation(128, 128, 2048, 4);
}

TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedSplitK5Test)
{
    // K=1920 for split_k=5: 1920/5=384=3×128 ✓
    // K must be divisible by K_Tile(128)*split_k(5)=640
    this->run_test_with_validation(128, 128, 1920, 5);
}
