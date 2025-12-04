// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <gtest/gtest.h>
#include <memory>

#include "test_gemm_quant_fixtures.hpp"

// Type aliases for readability
using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor   = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8           = ck_tile::fp8_t;
using BF8           = ck_tile::bf8_t;
using Half          = ck_tile::half_t;
using PkInt4        = ck_tile::pk_int4_t;
using BQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::BQuantGrouped>;
using GroupSize     = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
using GroupSize64   = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;

// 2d block sizes for BQuant
using GroupSize2D8N   = ck_tile::QuantGroupShape<ck_tile::sequence<1, 8, 128>>;
using GroupSize2D16N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 16, 128>>;
using GroupSize2D32N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 32, 128>>;
using GroupSize2D64N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 64, 128>>;
using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// Type combinations for BQuant tests (without PreshuffleB)
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using BQuantTypes = ::testing::Types<
    // 1d cases with grouping only on k axis (AQLayout is always RowMajor for BQuant)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize>,

    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize64>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize64>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize64>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize64>,

    // 2d cases with grouping also on the n axis
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D8N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D8N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D8N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D8N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D16N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D16N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D16N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D16N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D32N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D32N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D32N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D32N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D64N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D128N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8,    float, Half, BQuantGrouped, GemmConfigBase, GroupSize2D128N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, PkInt4, FP8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D128N>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, PkInt4, BF8,   Half, BQuantGrouped, GemmConfigBase, GroupSize2D128N>
>;
// clang-format on

// Test suite for BQuant (without PreshuffleB)
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuantTypes);

// BQuant tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
