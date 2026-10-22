// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompTDMProducerConsumerWmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMProducerConsumerWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompTDMProducerConsumerWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMProducerConsumerWmma,
                 KernelTypesCompTDMProducerConsumerWmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
