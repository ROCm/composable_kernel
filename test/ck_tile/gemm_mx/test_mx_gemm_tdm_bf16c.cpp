// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief MX GEMM with bf16 C on the gfx1250 TDM pipelines, checked with the tolerance derived in
 *        test_mx_gemm_host_helpers.hpp (one bf16 ULP plus the fp32 accumulation bound).
 */

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmTdmBf16C
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmBf16C<T>>
{
    public:
    static constexpr bool kUseDerivedMxTolerance = true;

    // Every row is chosen to run; an unsupported combination must not turn into a silent skip.
    static_assert(
        TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmBf16C<T>>::check_data_type(),
        "every bf16-C row must have WMMA traits");
};

#define TEST_SUITE_NAME TestCkTileMxGemmTdmBf16C

TYPED_TEST_SUITE(TestCkTileMxGemmTdmBf16C, KernelTypesMxGemmTdmBf16C);

#include "test_mx_gemm_pipeline_smoke_cases.inc"

#undef TEST_SUITE_NAME
