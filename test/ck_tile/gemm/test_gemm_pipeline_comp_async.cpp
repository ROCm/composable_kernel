// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompAsync
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompAsync<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompAsync

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompAsync);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME

template <typename T>
class TestCkTileGemmPipelineCompAsync16x16x128
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompAsync16x16x128<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompAsync16x16x128, KernelTypesCompAsync16x16x128);
TYPED_TEST(TestCkTileGemmPipelineCompAsync16x16x128, QuickTest)
{
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    this->template RunSingle<false, false, false, false>(M, N, K, 0, 0, 0, 1);
}
