// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_v3 bf16 gtest, short shapes (SMOKE_TEST): one, two and three K tiles and 512x512x1024 in
// every A/B layout; split-K: which k_batch the kernel accepts, and k_batch 2, 3 and 4 over a
// partial last K tile; and batched GEMM (V3BatchedConfigs, BatchedGemmKernel).
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

// Batched GEMM: 2 x 3 tiles per batch, each batch with its own A, B and C.
template <typename Config>
class TestCkTileGemmPipelineCompTDMV3Batched : public TestCkTileGemmPipelineCompTDMV3<Config>
{
};
TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMV3Batched, V3BatchedConfigs);

TYPED_TEST(TestCkTileGemmPipelineCompTDMV3Batched, IntBatched)
{
    this->RunBatched(
        2 * TypeParam::TileM, 3 * TypeParam::TileN, 2 * TypeParam::TileK, Init::Int, 3);
}
TYPED_TEST(TestCkTileGemmPipelineCompTDMV3Batched, RandomBatched)
{
    this->RunBatched(TypeParam::TileM, TypeParam::TileN, 3 * TypeParam::TileK, Init::Random, 5);
}

} // namespace
