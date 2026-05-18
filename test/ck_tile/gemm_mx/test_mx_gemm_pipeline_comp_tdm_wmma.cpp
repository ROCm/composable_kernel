// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmPipelineCompTDMWmma
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmPipelineCompTDMWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileMxGemmPipelineCompTDMWmma

TYPED_TEST_SUITE(TestCkTileMxGemmPipelineCompTDMWmma, KernelTypesMxGemmCompTDMWmma);

#include "test_mx_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
