// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using AQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>;

// Type combinations for AQuant tests - RRR and CRR layouts
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using AQuantBaseRRRCRRTypes = ::testing::Types<
    // RRR layout (RowMajor A, RowMajor B, RowMajor C with RowMajor AQ)
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize1D_128>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize1D_128>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize1D_128>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigBase, GroupSize1D_128>,

    // CRR layout (ColumnMajor A, RowMajor B, RowMajor C with RowMajor AQ)
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize1D_128>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize1D_128>
>;
// clang-format on

// Test suite for AQuant Base RRR/CRR
TYPED_TEST_SUITE(TestCkTileGemmAQuant, AQuantBaseRRRCRRTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmAQuant, AQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
