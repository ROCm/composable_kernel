// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_mx_grouped_gemm_util.hpp"
#include "test_mx_grouped_gemm_pipeline_kernel_types.hpp"

template <typename T>
class TestCkTileMxGemmPipelineCompAsync
    : public TestCkTileMxGroupedGemm<T, TestCkTileMxGemmPipelineCompAsync<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

#define TEST_SUITE_NAME TestCkTileMxGemmPipelineCompAsync

TYPED_TEST_SUITE(TestCkTileMxGemmPipelineCompAsync, KernelTypesMxGemmCompAsync);

#include "test_mx_grouped_gemm_ut_cases.inc"

#undef TEST_SUITE_NAME
