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

// TdmEpilogue overwrites C, so this pairing must keep rejecting split-K on the host; the
// CShuffleEpilogue twin (test_gemm_pipeline_comp_tdm_producer_consumer_cshuffle_wmma.cpp) is the
// only split-K path for this pipeline. Host-only: nothing is allocated or launched.
TYPED_TEST(TestCkTileGemmPipelineCompTDMProducerConsumerWmma, TdmEpilogueRejectsSplitK)
{
    if constexpr(TestFixture::check_data_type())
    {
        constexpr int M = TestFixture::M_Tile;
        constexpr int N = TestFixture::N_Tile;
        // Whole K tiles for every k_batch below, so only the epilogue can reject the split.
        constexpr int K = 4 * TestFixture::K_Tile;
        const auto stride_A =
            ck_tile::get_default_stride(M, K, 0, is_row_major(typename TestFixture::ALayout{}));
        const auto stride_B =
            ck_tile::get_default_stride(K, N, 0, is_row_major(typename TestFixture::BLayout{}));
        const auto stride_C =
            ck_tile::get_default_stride(M, N, 0, is_row_major(typename TestFixture::CLayout{}));

        EXPECT_TRUE(TestFixture::IsArgSupported(M, N, K, stride_A, stride_B, stride_C, 1));
        for(int k_batch : {2, 4})
        {
            EXPECT_FALSE(
                TestFixture::IsArgSupported(M, N, K, stride_A, stride_B, stride_C, k_batch))
                << "k_batch=" << k_batch;
        }
    }
}

#undef TEST_SUITE_NAME

// TdmEpilogue stores C with a plain TDM store, so a split-K launch would overwrite C rather than
// sum the splits. Half-precision output was already rejected through GetVectorSizeC(); fp32 output
// must be rejected too. Argument validation only: nothing is allocated or launched.
TEST(TestCkTileGemmPipelineCompTDMProducerConsumerWmmaArgs, SplitKRejectedForFloatOutput)
{
    // IsSupportedArgument rejects every problem without named barriers, so gate like the fixture.
    CK_TILE_SKIP_UNLESS_GFX125();

    using Fixture =
        TestCkTileGemmPipelineCompTDMProducerConsumerWmma<std::tuple<Row,
                                                                     Col,
                                                                     Row,
                                                                     F16,
                                                                     F16,
                                                                     F32,
                                                                     F32,
                                                                     I64,
                                                                     I64,
                                                                     I32,
                                                                     I16,
                                                                     I16,
                                                                     Intrawave,
                                                                     CompTDMProducerConsumer>>;
    constexpr ck_tile::index_t M = 64;
    constexpr ck_tile::index_t N = 64;
    constexpr ck_tile::index_t K = 1024;

    EXPECT_TRUE(Fixture::IsArgSupported(M, N, K, K, K, N, 1));
    EXPECT_FALSE(Fixture::IsArgSupported(M, N, K, K, K, N, 2));
}
