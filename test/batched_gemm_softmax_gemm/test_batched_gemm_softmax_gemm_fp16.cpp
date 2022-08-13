// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "test_batched_gemm_softmax_gemm_util.hpp"

template <typename Tuple>
class TestBatchedGemmSoftmaxGemmFP16 : public TestBatchedGemmSoftmaxGemm<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<F16, F16, F16, F16, Row, Col, Row, Row>
    >;
// clang-format on

TYPED_TEST_SUITE(TestBatchedGemmSoftmaxGemmFP16, KernelTypes);

TYPED_TEST(TestBatchedGemmSoftmaxGemmFP16, Test_FP16) { this->Run(); }

TYPED_TEST(TestBatchedGemmSoftmaxGemmFP16, DISABLED_Bench_FP16)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {256, 256, 64, 64, 768},
        {256, 256, 128, 128, 768},
        {512, 512, 64, 64, 768},
        {512, 512, 128, 128, 768},
        {1024, 1024, 64, 64, 768},
        {1024, 1024, 128, 128, 768},
        {2048, 2048, 64, 64, 768},
        {2048, 2048, 128, 128, 768},
        {4096, 4096, 64, 64, 768},
        {4096, 4096, 128, 128, 768},
    };
    this->bench_  = true;
    this->verify_ = false;
    this->Run();
}
