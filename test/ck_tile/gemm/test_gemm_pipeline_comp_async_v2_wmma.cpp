// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompAsyncV2Wmma
    : public TestCkTileGemmPipelineWmmaBase<T, class TestCkTileGemmPipelineCompAsyncV2Wmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompAsyncV2Wmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompAsyncV2Wmma, KernelTypesCompAsyncV2Wmma);

#include "test_gemm_pipeline_comp_async_v2_tail_cases.inc"

#undef TEST_SUITE_NAME
