// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"

#include "mx_mfma_op.hpp"

using ck::e8m0_bexp_t;
using ck::f8_t;
using ck::half_t;
using ck::type_convert;

template <typename AType, typename BType, typename CType, ck::mx_mfma_test::MFMA_F8F6F4 mfma>
bool run_test()
{
    using ALayout = ck::tensor_layout::gemm::ColumnMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::ColumnMajor;

    using AccType    = float; // only MFMA_F32 instructions supported
    using CPUAccType = AccType;

    ck::mfma_type<static_cast<ck::MfmaInstr>(mfma)> mfma_instr;
    constexpr auto BLOCK_M = mfma_instr.m_per_blk;
    constexpr auto BLOCK_N = mfma_instr.n_per_blk;
    constexpr auto BLOCK_K = mfma_instr.num_input_blks * mfma_instr.k_per_blk;

    const auto mx_mfma_kernel =
        ck::mx_mfma_test::matmul<AType, BType, CType, AccType, BLOCK_M, BLOCK_N, BLOCK_K>;

    bool pass = true;

    pass = ck::mx_mfma_test::TestMFMA<decltype(mx_mfma_kernel),
                                      AType,
                                      BType,
                                      CType,
                                      AccType,
                                      CPUAccType,
                                      ALayout,
                                      BLayout,
                                      CLayout,
                                      BLOCK_M,
                                      BLOCK_N,
                                      BLOCK_K>{}(mx_mfma_kernel);

    return pass;
}

TEST(MFMA, FP8MFMA16x16x128)
{
    auto pass = run_test<f8_t, f8_t, half_t, ck::mx_mfma_test::MFMA_F8F6F4::F32_16x16x128>();
    EXPECT_TRUE(pass);
}

TEST(MFMA, FP8MFMA32x32x64)
{
    auto pass = run_test<f8_t, f8_t, float, ck::mx_mfma_test::MFMA_F8F6F4::F32_32x32x64>();
    EXPECT_TRUE(pass);
}

// TEST(MXMFMA, FP8MFMA32x32x64)
// {
//     EXPECT_TRUE(run_test<f8, 1, f8, 1, float, 1, float, float, 32, 32, 64>());
// }

// TEST(MXMFMA, BF8MFMA16x16x128)
// {
//     EXPECT_TRUE(run_test<bf8, 1, bf8, 1, float, 1, float, float, 16, 16, 128>());
// }

// TEST(MXMFMA, BF8MFMA32x32x64)
// {
//     EXPECT_TRUE(run_test<bf8, 1, bf8, 1, float, 1, float, float, 32, 32, 64>());
// }

// TEST(MXMFMA, MXFP8xMXFP8) { EXPECT_TRUE(false) << "Not Implemented\n"; }
// TEST(MXMFMA, MXBF8xMXBF8) { EXPECT_TRUE(false) << "Not Implemented\n"; }
