// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Large-tensor / decomposition cases for the fp4 (a4w4) grouped MX GEMM (ROCM-22075).
//
// This is a SEPARATE executable from test_ck_tile_grouped_gemm_mx_comp_async and is intentionally
// NOT registered with ctest (its CMake target uses add_executable, not add_gtest_executable), so
// it is excluded from the default CI test pass. It is run explicitly (mirroring the CK
// *_large_cases / RUN_*_LARGE_CASES_TESTS convention) because it allocates multi-GB device buffers
// (per-group C ~2.5 GB) to exercise the int32 element-count / decomposition boundary.

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

#include "test_mx_grouped_gemm_largeM_cases.inc"

#undef TEST_SUITE_NAME
