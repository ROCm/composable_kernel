// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "mx_wmma_op_test_common.hpp"

// =============================================================================
// MXWMMA_ISOSCALE: FP4 companions (init=3 all-ones scale,
//                                  init=4 all-zeros scale)
// =============================================================================

// --- FP4@FP4, e8m0, block size 32 ---
TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP4@FP4, e8m0, block size 16 ---
TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_SCALE16_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_SCALE16_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP4@FP4, e4m3, block size 32 ---
TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_E4M3_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e4m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_E4M3_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e4m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP4@FP4, e4m3, block size 16 ---
TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_SCALE16_E4M3_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e4m3_scale_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_SCALE16_E4M3_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e4m3_scale_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP4@FP4, e5m3, block size 32 ---
TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_E5M3_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e5m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_E5M3_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e5m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP4@FP4, e5m3, block size 16 ---
TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_SCALE16_E5M3_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e5m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_SCALE16_E5M3_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e5m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP4@FP4, e4m3 A-scale + e5m3 B-scale, block size 32 ---
TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_E4M3_E5M3_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_E4M3_E5M3_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP4@FP4, e4m3 A-scale + e5m3 B-scale, block size 16 ---
TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_SCALE16_E4M3_E5M3_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_SCALE16_E4M3_E5M3_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP4@FP4, e5m3 A-scale + e4m3 B-scale, block size 32 ---
TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_E5M3_E4M3_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e5m3_scale_t,
                                 e4m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP4WMMA16x16x128_E5M3_E4M3_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e5m3_scale_t,
                                 e4m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}
