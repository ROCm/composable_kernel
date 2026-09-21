// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "mx_wmma_op_test_common.hpp"

// =============================================================================
// MXWMMA_ISOSCALE: FP6 / BF6 companions (init=3 all-ones scale,
//                                        init=4 all-zeros scale)
// =============================================================================

// --- FP6@FP6, e8m0, block size 32 ---
TEST(MXWMMA_ISOSCALE, MXFP6WMMA16x16x128_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f6_t,
                                 f6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP6WMMA16x16x128_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f6_t,
                                 f6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- FP6@FP6, e8m0, block size 16 ---
TEST(MXWMMA_ISOSCALE, MXFP6WMMA16x16x128_SCALE16_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f6_t,
                                 f6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXFP6WMMA16x16x128_SCALE16_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f6_t,
                                 f6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- BF6@BF6, e8m0, block size 32 ---
TEST(MXWMMA_ISOSCALE, MXBF6WMMA16x16x128_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf6_t,
                                 bf6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXBF6WMMA16x16x128_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf6_t,
                                 bf6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}

// --- BF6@BF6, e8m0, block size 16 ---
TEST(MXWMMA_ISOSCALE, MXBF6WMMA16x16x128_SCALE16_E8M0_SCALE_1)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf6_t,
                                 bf6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(3);
    EXPECT_TRUE(pass);
}

TEST(MXWMMA_ISOSCALE, MXBF6WMMA16x16x128_SCALE16_E8M0_SCALE_0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf6_t,
                                 bf6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(4);
    EXPECT_TRUE(pass);
}
