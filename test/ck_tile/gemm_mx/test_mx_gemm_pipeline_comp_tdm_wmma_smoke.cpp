// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief The TDM rows of test_mx_gemm_pipeline_comp_tdm_wmma.cpp with the cases of the gfx1250
 *        emulator smoke stage, whose shapes differ from the full binary's.
 */

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmPipelineCompTDMWmmaSmoke
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmPipelineCompTDMWmmaSmoke<T>>
{
};

#define TEST_SUITE_NAME TestCkTileMxGemmPipelineCompTDMWmmaSmoke

TYPED_TEST_SUITE(TestCkTileMxGemmPipelineCompTDMWmmaSmoke, KernelTypesMxGemmCompTDMWmma);

#include "test_mx_gemm_pipeline_smoke_cases.inc"

#undef TEST_SUITE_NAME
