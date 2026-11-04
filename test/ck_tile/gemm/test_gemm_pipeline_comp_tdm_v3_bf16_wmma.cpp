// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_v3 bf16 gtest, short shapes (SMOKE_TEST): one, two and three K tiles and 512x512x1024 in
// every A/B layout; and split-K: which k_batch the kernel accepts, and k_batch 2, 3 and 4 over a
// partial last K tile.
// Instances (V3Bf16Configs), inputs and checks: test_gemm_pipeline_comp_tdm_v3_util.hpp.

#include "test_gemm_pipeline_comp_tdm_v3_util.hpp"

namespace {

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMV3, V3Bf16Configs);

TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, IntOneTwoThreeKTiles)
{
    for(ck_tile::index_t tiles : {1, 2, 3})
        this->Run(256, 256, tiles * TypeParam::TileK, Init::Int);
}
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, Int512) { this->Run(512, 512, 1024, Init::Int); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, RandomOneTwoThreeKTiles)
{
    for(ck_tile::index_t tiles : {1, 2, 3})
        this->Run(256, 256, tiles * TypeParam::TileK, Init::Random);
}
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, Random512) { this->Run(512, 512, 1024, Init::Random); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, SplitKSupport) { this->CheckSplitKSupport(); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3, SplitK) { this->RunSplitK(); }

} // namespace
