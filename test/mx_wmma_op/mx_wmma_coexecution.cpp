// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include "gtest/gtest.h"

#include "mx_wmma_coexecution.hpp"

using ck::e4m3_scale_t;
using ck::e5m3_scale_t;
using ck::e8m0_bexp_t;
using ck::f4_t;
using ck::f4x2_pk_t;
using ck::type_convert;

/**
 * @brief Run the test for the given WMMA scale instruction
 *
 * @param init - selects initialization algorithm for A and B tensors
 */
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AType,
          typename BType,
          typename CType,
          typename AScaleType,
          typename BScaleType,
          ck::WMMA_SCALE wmma,
          ck::index_t num_steps>
bool run_mx_wmma_coexecution_test(ck::index_t init)
{
    static_assert((wmma == ck::WMMA_SCALE::SCALE_F32_16x16x128) ||
                      (wmma == ck::WMMA_SCALE::SCALE16_F32_16x16x128) ||
                      (wmma == ck::WMMA_SCALE::SCALE_F32_32x16x128) ||
                      (wmma == ck::WMMA_SCALE::SCALE16_F32_32x16x128),
                  "Only SCALE_F32_16x16x128, SCALE16_F32_16x16x128, SCALE_F32_32x16x128, and "
                  "SCALE16_F32_32x16x128 are supported");

    using AccType = float; // only F32 instructions supported

    // WMMA scale instruction parameters
    ck::mfma_type<static_cast<ck::MfmaInstr>(wmma)> wmma_instr;
    constexpr auto BLOCK_M = wmma_instr.m_per_blk;
    constexpr auto BLOCK_N = wmma_instr.n_per_blk;
    constexpr auto BLOCK_K = wmma_instr.num_input_blks * wmma_instr.k_per_blk;
    constexpr auto BLOCK_X = wmma_instr.scale_blk_size; // scaling vector size

    const auto mx_wmma_kernel = ck::matmul<AType,
                                           BType,
                                           AScaleType,
                                           BScaleType,
                                           CType,
                                           AccType,
                                           BLOCK_M,
                                           BLOCK_N,
                                           BLOCK_K,
                                           BLOCK_X,
                                           ALayout,
                                           BLayout,
                                           CLayout,
                                           num_steps>;

    bool pass = true;

    pass = ck::mx_wmma_test::TestMXWMMA<decltype(mx_wmma_kernel),
                                        AType,
                                        BType,
                                        AScaleType,
                                        BScaleType,
                                        CType,
                                        ALayout,
                                        BLayout,
                                        CLayout,
                                        BLOCK_M,
                                        BLOCK_N,
                                        BLOCK_K,
                                        BLOCK_X>{}(mx_wmma_kernel, init);

    return pass;
}

const ck::index_t common_init = -1;

// test FP4@FP4 with 16x16x128 instruction, scale block size 32, and e8m0 scales
TEST(MXWMMA, MXFP4WMMA16x16x128_E8M0)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 1;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e8m0_bexp_t,
                                             e8m0_bexp_t,
                                             ck::WMMA_SCALE::SCALE_F32_16x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 32, and e8m0 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_E8M0)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e8m0_bexp_t,
                                             e8m0_bexp_t,
                                             ck::WMMA_SCALE::SCALE_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 16x16x128 instruction, scale block size 16, and e8m0 scales
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E8M0)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 1;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e8m0_bexp_t,
                                             e8m0_bexp_t,
                                             ck::WMMA_SCALE::SCALE16_F32_16x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}
// test FP4@FP4 with 32x16x128 instruction, scale block size 16, and e8m0 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_SCALE16_E8M0)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e8m0_bexp_t,
                                             e8m0_bexp_t,
                                             ck::WMMA_SCALE::SCALE16_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 32, and e4m3 scale type
TEST(MXWMMA, MXFP4WMMA32x16x128_E4M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e4m3_scale_t,
                                             e4m3_scale_t,
                                             ck::WMMA_SCALE::SCALE_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 16, and e4m3 scale type
TEST(MXWMMA, MXFP4WMMA32x16x128_SCALE16_E4M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e4m3_scale_t,
                                             e4m3_scale_t,
                                             ck::WMMA_SCALE::SCALE16_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 32, and e5m3 scale type
TEST(MXWMMA, MXFP4WMMA32x16x128_E5M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e5m3_scale_t,
                                             e5m3_scale_t,
                                             ck::WMMA_SCALE::SCALE_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 16, and e5m3 scale type
TEST(MXWMMA, MXFP4WMMA32x16x128_SCALE16_E5M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e5m3_scale_t,
                                             e5m3_scale_t,
                                             ck::WMMA_SCALE::SCALE16_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 32, and e8m0 and e4m3 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_E8M0_E4M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e8m0_bexp_t,
                                             e4m3_scale_t,
                                             ck::WMMA_SCALE::SCALE_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 16, and e8m0 and e4m3 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_SCALE16_E8M0_E4M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e8m0_bexp_t,
                                             e4m3_scale_t,
                                             ck::WMMA_SCALE::SCALE16_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 32, and e8m0 and e5m3 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_E8M0_E5M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e8m0_bexp_t,
                                             e5m3_scale_t,
                                             ck::WMMA_SCALE::SCALE_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 16, and e8m0 and e5m3 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_SCALE16_E8M0_E5M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e8m0_bexp_t,
                                             e5m3_scale_t,
                                             ck::WMMA_SCALE::SCALE16_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 32, and e4m3 and e8m0 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_E4M3_E8M0)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e4m3_scale_t,
                                             e8m0_bexp_t,
                                             ck::WMMA_SCALE::SCALE_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 16, and e4m3 and e8m0 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_SCALE16_E4M3_E8M0)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e4m3_scale_t,
                                             e8m0_bexp_t,
                                             ck::WMMA_SCALE::SCALE16_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 32, and e4m3 and e5m3 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_E4M3_E5M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e4m3_scale_t,
                                             e5m3_scale_t,
                                             ck::WMMA_SCALE::SCALE_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 16, and e4m3 and e5m3 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_SCALE16_E4M3_E5M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e4m3_scale_t,
                                             e5m3_scale_t,
                                             ck::WMMA_SCALE::SCALE16_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 32, and e5m3 and e8m0 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_E5M3_E8M0)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e5m3_scale_t,
                                             e8m0_bexp_t,
                                             ck::WMMA_SCALE::SCALE_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 16, and e5m3 and e8m0 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_SCALE16_E5M3_E8M0)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e5m3_scale_t,
                                             e8m0_bexp_t,
                                             ck::WMMA_SCALE::SCALE16_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 32, and e5m3 and e4m3 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_E5M3_E4M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e5m3_scale_t,
                                             e4m3_scale_t,
                                             ck::WMMA_SCALE::SCALE_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with 32x16x128 instruction, scale block size 16, and e5m3 and e4m3 scales
TEST(MXWMMA, MXFP4WMMA32x16x128_SCALE16_E5M3_E4M3)
{
    if(ck::get_device_revision() == 0)
    {
        GTEST_SKIP() << "This WMMA is not supported on asicRevision=0";
    }

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    const ck::index_t num_steps = 2;

    auto pass = run_mx_wmma_coexecution_test<ALayout,
                                             BLayout,
                                             CLayout,
                                             f4_t,
                                             f4_t,
                                             float,
                                             e5m3_scale_t,
                                             e4m3_scale_t,
                                             ck::WMMA_SCALE::SCALE16_F32_32x16x128,
                                             num_steps>(common_init);
    EXPECT_TRUE(pass);
}
