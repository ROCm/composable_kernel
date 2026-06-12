// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// Type combinations for ABQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantPreshuffleQuantTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigPreshuffleBPreshuffleQuantPrefill<false>, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigPreshuffleBPreshuffleQuantPrefill<true>, GroupSize1D_128, GroupSize2D128N, ColumnMajor>
>;
// clang-format on

// Test suite for ABQuant
TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantPreshuffleQuantTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest)
{
    using BQuantGroupSize = std::tuple_element_t<11, TypeParam>;
    if(ck_tile::is_gfx120_supported() && std::is_same_v<BQuantGroupSize, GroupSize2D128N>)
    {
        GTEST_SKIP() << "temp disable due to random fail on gfx120.";
    }

    this->run_test_with_validation(1024, 1024, 1024);
}
