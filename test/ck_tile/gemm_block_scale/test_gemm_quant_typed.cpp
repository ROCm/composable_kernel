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
using GroupSize     = std::integral_constant<unsigned int, 128>;

// Type combinations for each quantization type
// clang-format off
using AQuantTypes = ::testing::Types<
    // PreshuffleQuant = false && TransposeC = false
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigBase, GroupSize>,

    // PreshuffleQuant = false && TransposeC = true 
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigTransposeC, GroupSize>,

    // PreshuffleQuant = true && TransposeC = false
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigPreshuffleQuant, GroupSize>,

    // PreshuffleQuant = true && TransposeC = true 
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, FP8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, BF8, float, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, PkInt4, FP8, FP8, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, PkInt4, BF8, BF8, Half, AQuantGrouped, GemmConfigPreshuffleQuantTransposeC, GroupSize>
>;
// clang-format on

// clang-format off
using BQuantTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, FP8, float, Half, BQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, BF8, float, Half, BQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, PkInt4, BF8, Half, BQuantGrouped, GemmConfigBase, GroupSize>
>;
// clang-format on

// clang-format off
using BPreshuffleBQuantTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, FP8, float, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, BF8, float, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, PkInt4, BF8, Half, BQuantGrouped, GemmConfigPreshuffleBDecode, GroupSize>,

    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, FP8, float, Half, BQuantGrouped, GemmConfigPreshuffleBPrefill, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, BF8, float, Half, BQuantGrouped, GemmConfigPreshuffleBPrefill, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, PkInt4, FP8, Half, BQuantGrouped, GemmConfigPreshuffleBPrefill, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, PkInt4, BF8, Half, BQuantGrouped, GemmConfigPreshuffleBPrefill, GroupSize>
>;

// clang-format off
using RowColQuantTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, BF8, float, Half, RowColQuant, GemmConfigBase, GroupSize>
>;
// clang-format on

// clang-format off
using TensorQuantTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, FP8, FP8, float, Half, TensorQuant, GemmConfigBase, GroupSize>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, BF8, BF8, float, Half, TensorQuant, GemmConfigBase, GroupSize>
>;
// clang-format on

// Test suites for each quantization type
TYPED_TEST_SUITE(TestCkTileGemmAQuant, AQuantTypes);
TYPED_TEST_SUITE(TestCkTileGemmBQuant, BQuantTypes);
TYPED_TEST_SUITE(TestCkTileGemmPreshuffleBBQuant, BPreshuffleBQuantTypes);
TYPED_TEST_SUITE(TestCkTileGemmRowColQuant, RowColQuantTypes);
TYPED_TEST_SUITE(TestCkTileGemmTensorQuant, TensorQuantTypes);

#include "test_gemm_quant_ut_cases.inc"
