// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompAsyncV2
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompAsyncV2<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompAsyncV2

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompAsyncV2);

#include "test_gemm_pipeline_comp_async_v2_tail_cases.inc"

#undef TEST_SUITE_NAME
