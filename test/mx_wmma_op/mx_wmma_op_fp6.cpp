// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "mx_wmma_op_test_common.hpp"

// test FP6@FP6 with e8m0 scale and 32 block size
TEST(MXWMMA, MXFP6WMMA16x16x128_E8M0)
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
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP6@FP6 with e8m0 scale and 16 block size
TEST(MXWMMA, MXFP6WMMA16x16x128_SCALE16_E8M0)
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
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test BF6@BF6 with e8m0 scale and 32 block size
TEST(MXWMMA, MXBF6WMMA16x16x128_E8M0)
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
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test BF6@BF6 with e8m0 scale and 16 block size
TEST(MXWMMA, MXBF6WMMA16x16x128_SCALE16_E8M0)
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
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}
