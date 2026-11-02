// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_v3 gtest, short shapes (SMOKE_TEST): one, two and three K tiles and 512x512x1024; and
// the split-K rejection.
// Instances, inputs and checks: test_gemm_pipeline_comp_tdm_v3_util.hpp.

#include "test_gemm_pipeline_comp_tdm_v3_util.hpp"

namespace {

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMV3, V3Configs);

TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, IntOneTwoThreeKTiles)
{
    for(ck_tile::index_t k : {256, 512, 768})
        this->Run(256, 256, k, Init::Int);
}
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, Int512) { this->Run(512, 512, 1024, Init::Int); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, RandomOneTwoThreeKTiles)
{
    for(ck_tile::index_t k : {256, 512, 768})
        this->Run(256, 256, k, Init::Random);
}
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, Random512) { this->Run(512, 512, 1024, Init::Random); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, SplitKRejected) { this->CheckSplitKRejected(); }

} // namespace
