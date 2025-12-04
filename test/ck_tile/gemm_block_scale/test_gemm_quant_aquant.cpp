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
using AQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>;
using BQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::BQuantGrouped>;
using RowColQuant   = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::RowColQuant>;
using TensorQuant   = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::TensorQuant>;
using GroupSize     = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
using GroupSize64   = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;

// 2d block sizes for BQuant
using GroupSize2D8N   = ck_tile::QuantGroupShape<ck_tile::sequence<1, 8, 128>>;
using GroupSize2D16N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 16, 128>>;
using GroupSize2D32N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 32, 128>>;
using GroupSize2D64N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 64, 128>>;
using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// Type combinations for AQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using AQuantTypes = ::testing::Types<
    // PreshuffleQuant = false && TransposeC = false (RCR layout with RowMajor AQ)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,

    // RRR layout (RowMajor A, RowMajor B, RowMajor C with RowMajor AQ)
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,

    // CRR layout (ColumnMajor A, RowMajor B, RowMajor C with RowMajor AQ)
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,

    // CCR layout (ColumnMajor A, ColumnMajor B, RowMajor C with ColumnMajor AQ) - NEW layout support
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, ColumnMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,

    // RCR layout - with the Prefill BlockTile Config.
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigPrefill, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigPrefill, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigPrefill, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigPrefill, GroupSize>,

    // PreshuffleQuant = false && TransposeC = true (with RowMajor AQ)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigTransposeC, GroupSize>,

    // PreshuffleQuant = true && TransposeC = false (with RowMajor AQ - PreshuffleQuant only supports RowMajor)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,

    // PreshuffleQuant = true && TransposeC = true (with RowMajor AQ - PreshuffleQuant only supports RowMajor)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>
>;
// clang-format on

// Test suite for AQuant
TYPED_TEST_SUITE(TestCkTileGemmAQuant, AQuantTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmAQuant, AQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
