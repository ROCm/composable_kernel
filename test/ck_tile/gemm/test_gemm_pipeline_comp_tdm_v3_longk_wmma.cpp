// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_v3 gtest, K = 65536 (REGRESSION_TEST: too slow for the smoke set under the gfx1250
// emulator). Instances, inputs and checks: test_gemm_pipeline_comp_tdm_v3_util.hpp.

#include "test_gemm_pipeline_comp_tdm_v3_util.hpp"

namespace {

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMV3, V3LongKConfigs);

TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, IntLongK) { this->Run(256, 256, 65536, Init::Int); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, RandomLongK)
{
    this->Run(256, 256, 65536, Init::Random);
}

} // namespace
