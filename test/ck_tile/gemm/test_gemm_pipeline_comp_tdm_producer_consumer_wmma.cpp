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
