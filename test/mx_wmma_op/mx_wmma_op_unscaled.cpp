// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "mx_wmma_op_test_common.hpp"

// Unscaled WMMA: test wmma_f16_16x16x128_bf8_bf8_gfx125
TEST(MXWMMA_UNSCALED, MXF16WMMA16x16x128_BF8_BF8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::bf8_t;
    using BType   = ck::bf8_t;
    using CType   = ck::half_t;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 128;
    using AccType         = ck::half_t;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f16_16x16x128_bf8_fp8_gfx125
TEST(MXWMMA_UNSCALED, MXF16WMMA16x16x128_BF8_FP8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::bf8_t;
    using BType   = ck::f8_t;
    using CType   = ck::half_t;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 128;
    using AccType         = ck::half_t;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f16_16x16x128_fp8_bf8_gfx125
TEST(MXWMMA_UNSCALED, MXF16WMMA16x16x128_FP8_BF8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::f8_t;
    using BType   = ck::bf8_t;
    using CType   = ck::half_t;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 128;
    using AccType         = ck::half_t;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f16_16x16x128_fp8_fp8_gfx125
TEST(MXWMMA_UNSCALED, MXF16WMMA16x16x128_FP8_FP8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::f8_t;
    using BType   = ck::f8_t;
    using CType   = ck::half_t;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 128;
    using AccType         = ck::half_t;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f16_16x16x64_f8f8_gfx125
TEST(MXWMMA_UNSCALED, MXF16WMMA16x16x64_FP8_FP8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::f8_t;
    using BType   = ck::f8_t;
    using CType   = ck::half_t;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 64;
    using AccType         = ck::half_t;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f16_16x16x64_f8bf8_gfx125
TEST(MXWMMA_UNSCALED, MXF16WMMA16x16x64_FP8_BF8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::f8_t;
    using BType   = ck::bf8_t;
    using CType   = ck::half_t;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 64;
    using AccType         = ck::half_t;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f16_16x16x64_bf8f8_gfx125
TEST(MXWMMA_UNSCALED, MXF16WMMA16x16x64_BF8_FP8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::bf8_t;
    using BType   = ck::f8_t;
    using CType   = ck::half_t;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 64;
    using AccType         = ck::half_t;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f16_16x16x64_bf8bf8_gfx125
TEST(MXWMMA_UNSCALED, MXF16WMMA16x16x64_BF8_BF8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::bf8_t;
    using BType   = ck::bf8_t;
    using CType   = ck::half_t;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 64;
    using AccType         = ck::half_t;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f32_16x16x128_bf8_bf8_gfx125
TEST(MXWMMA_UNSCALED, MXF32WMMA16x16x128_BF8_BF8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::bf8_t;
    using BType   = ck::bf8_t;
    using CType   = float;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 128;
    using AccType         = float;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f32_16x16x128_bf8_fp8_gfx125
TEST(MXWMMA_UNSCALED, MXF32WMMA16x16x128_BF8_FP8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::bf8_t;
    using BType   = ck::f8_t;
    using CType   = float;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 128;
    using AccType         = float;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f32_16x16x128_fp8_bf8_gfx125
TEST(MXWMMA_UNSCALED, MXF32WMMA16x16x128_FP8_BF8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::f8_t;
    using BType   = ck::bf8_t;
    using CType   = float;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 128;
    using AccType         = float;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}

// Unscaled WMMA: test wmma_f32_16x16x128_fp8_fp8_gfx125
TEST(MXWMMA_UNSCALED, MXF32WMMA16x16x128_FP8_FP8_GFX125)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;
    using AType   = ck::f8_t;
    using BType   = ck::f8_t;
    using CType   = float;

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 128;
    using AccType         = float;

    auto pass = run_mx_wmma_unscaled_test<ALayout,
                                          BLayout,
                                          CLayout,
                                          AType,
                                          BType,
                                          CType,
                                          AccType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          BLOCK_K>(common_init);
    EXPECT_TRUE(pass);
}
