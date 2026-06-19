// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompV3AsyncCCR
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV3AsyncCCR<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
    static constexpr bool Async = true;
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompV3AsyncCCR

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompV3AsyncCCR);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
