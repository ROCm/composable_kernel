// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "mx_wmma_op_test_common.hpp"

// =============================================================================
// MXWMMA_ISOSCALE: FP8 / BF8 companions (init=3 all-ones scale,
//                                        init=4 all-zeros scale)
// =============================================================================

// --- FP8@FP8, e8m0, block size 32 ---
TEST(MXWMMA_ISOSCALE, MXFP8WMMA16x16x128_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP8WMMA16x16x128_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP8@FP8, e8m0, block size 16 ---
TEST(MXWMMA_ISOSCALE, MXFP8WMMA16x16x128_SCALE16_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP8WMMA16x16x128_SCALE16_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- BF8@BF8, e8m0, block size 32 ---
TEST(MXWMMA_ISOSCALE, MXBF8WMMA16x16x128_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf8_t,
                                 bf8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXBF8WMMA16x16x128_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf8_t,
                                 bf8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- BF8@BF8, e8m0, block size 16 ---
TEST(MXWMMA_ISOSCALE, MXBF8WMMA16x16x128_SCALE16_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf8_t,
                                 bf8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXBF8WMMA16x16x128_SCALE16_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf8_t,
                                 bf8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP8@FP4, e8m0, block size 32 ---
TEST(MXWMMA_ISOSCALE, MXFP8FP4WMMA16x16x128_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP8FP4WMMA16x16x128_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP8@FP4, e8m0, block size 16 ---
TEST(MXWMMA_ISOSCALE, MXFP8FP4WMMA16x16x128_SCALE16_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP8FP4WMMA16x16x128_SCALE16_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}
