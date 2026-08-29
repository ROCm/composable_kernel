// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "mx_wmma_op_test_common.hpp"

// test FP4@FP4 with e8m0 scale and 32 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_E8M0)
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
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e8m0 scale and 16 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E8M0)
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
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e4m3 scale and 32 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_E4M3)
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
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e4m3 scale and 16 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E4M3)
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
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e5m3 scale and 32 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_E5M3)
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
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e5m3 scale and 16 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E5M3)
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
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e4m3 and e5m3 scales and 32 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_E4M3_E5M3)
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
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e4m3 and e5m3 scales and 16 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E4M3_E5M3)
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
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e5m3 and e4m3 scales and 32 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_E5M3_E4M3)
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
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e5m3 and e4m3 scales and 16 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E5M3_E4M3)
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
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}
