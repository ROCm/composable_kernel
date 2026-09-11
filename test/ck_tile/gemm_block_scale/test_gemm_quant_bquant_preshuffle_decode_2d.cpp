// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize2D8N   = ck_tile::QuantGroupShape<ck_tile::sequence<1, 8, 128>>;
using GroupSize2D16N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 16, 128>>;
using GroupSize2D32N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 32, 128>>;
using GroupSize2D64N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 64, 128>>;
using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// Type combinations for BQuant Preshuffle tests - Decode 2D
// Tuple format: <ALayout, BLayout, CLayout, BQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BPreshuffleDecode2DTypes = ::testing::Types<
    // 2d cases with preshuffle B
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize2D8N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize2D8N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize2D16N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize2D16N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize2D32N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize2D32N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize2D64N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize2D64N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize2D128N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize2D128N>
>;
// clang-format on

// Test suite for BQuant Preshuffle Decode 2D
TYPED_TEST_SUITE(TestCkTileGemmPreshuffleBBQuant, BPreshuffleDecode2DTypes);

// BQuant PreshuffleB tests
TYPED_TEST(TestCkTileGemmPreshuffleBBQuant, BQuantPreshuffleTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
