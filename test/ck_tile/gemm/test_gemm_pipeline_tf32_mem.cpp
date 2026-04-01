// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineTf32Mem
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineTf32Mem<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineTf32Mem

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesTf32Mem);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
