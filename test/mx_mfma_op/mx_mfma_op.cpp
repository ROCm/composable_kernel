// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"

#include "mx_mfma_op.hpp"

using ck::e8m0_bexp_t;
using ck::f8_ocp_t;
using ck::type_convert;

template <typename Src1Type,
          ck::index_t Src1VecSize,
          typename Src2Type,
          ck::index_t Src2VecSize,
          typename DstType,
          ck::index_t AccVecSize,
          typename AccType,
          typename CPUAccType,
          ck::index_t M,
          ck::index_t N,
          ck::index_t K>
bool run_test()
{
    using Row         = ck::tensor_layout::gemm::RowMajor;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    bool pass         = true;

    const auto mx_mfma_kernel = ck::mx_mfma_test::
        matmul<Src1Type, Src1VecSize, Src2Type, Src2VecSize, AccType, AccVecSize, DstType, M, N, K>;

    pass = ck::mx_mfma_test::TestMXMFMA<decltype(mx_mfma_kernel),
                                        Src1Type,
                                        Src2Type,
                                        DstType,
                                        AccType,
                                        CPUAccType,
                                        decltype(Row{}),
                                        decltype(Row{}),
                                        decltype(Row{}),
                                        PassThrough,
                                        PassThrough,
                                        PassThrough,
                                        AccVecSize,
                                        M,
                                        N,
                                        K>{}(mx_mfma_kernel);

    return pass;
}

TEST(MXMFMA, FP8MFMA16x16x128)
{
    auto pass = run_test<float, 1, float, 1, float, 1, float, float, 16, 16, 128>();
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

TEST(MXMFMA, MXFP8xMXFP8) { EXPECT_TRUE(false) << "Not Implemented\n"; }
TEST(MXMFMA, MXBF8xMXBF8) { EXPECT_TRUE(false) << "Not Implemented\n"; }
