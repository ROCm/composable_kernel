// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmPipelineCompAsyncRCR
    : public TestCkTileMxGemmPipeline<T, TestCkTileMxGemmPipelineCompAsyncRCR<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

#define TEST_SUITE_NAME TestCkTileMxGemmPipelineCompAsyncRCR

TYPED_TEST_SUITE(TestCkTileMxGemmPipelineCompAsyncRCR, KernelTypesMxGemmCompAsyncRCRLargeCases);

TYPED_TEST(TEST_SUITE_NAME, Large)
{
    int M = 6422528;
    int N = 6144;
    int K = 1024;

    this->RunAllGpu(M, N, K);
}

#undef TEST_SUITE_NAME
