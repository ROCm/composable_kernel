// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompTDMRunAheadWmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMRunAheadWmma<T>>
{
    using Base = TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMRunAheadWmma<T>>;

    // The gate below is only right for the run-ahead pipeline, which exists on gfx1250 alone.
    static_assert(Base::PipelineType == GemmPipelineType::CompTDMRunAhead);

    protected:
    // gfx1250-only explicitly, so the suite does not depend on the shared fixture's TDM list or
    // on the kernel's runtime rejection. It also covers the host-only cases, whose kernels are
    // only built for gfx125. CK_TILE_REQUIRE_GFX125=1 (set by CMake for gfx125-only builds)
    // turns the skip into a failure on a runner that misreports its device.
    void SetUp() override
    {
        CK_TILE_SKIP_UNLESS_GFX125();
        Base::SetUp();
    }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompTDMRunAheadWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMRunAheadWmma, KernelTypesCompTDMRunAheadWmma);

#include "test_gemm_pipeline_ut_cases.inc"
#include "test_gemm_pipeline_comp_tdm_run_ahead_cases.inc"

#undef TEST_SUITE_NAME
