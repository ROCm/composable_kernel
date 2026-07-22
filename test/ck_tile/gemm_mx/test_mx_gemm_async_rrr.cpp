// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmPipelineCompAsyncRRR
    : public TestCkTileMxGemmPipeline<T, TestCkTileMxGemmPipelineCompAsyncRRR<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

#define TEST_SUITE_NAME TestCkTileMxGemmPipelineCompAsyncRRR

TYPED_TEST_SUITE(TestCkTileMxGemmPipelineCompAsyncRRR, KernelTypesMxGemmCompAsyncRRR);

#include "test_mx_gemm_pipeline_tr_cases.inc"

#undef TEST_SUITE_NAME
