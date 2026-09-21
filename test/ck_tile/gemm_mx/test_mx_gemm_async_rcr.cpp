// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmPipelineCompAsyncRCR
    : public TestCkTileMxGemmPipeline<T, TestCkTileMxGemmPipelineCompAsyncRCR<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

#define TEST_SUITE_NAME TestCkTileMxGemmPipelineCompAsyncRCR

TYPED_TEST_SUITE(TestCkTileMxGemmPipelineCompAsyncRCR, KernelTypesMxGemmCompAsyncRCR);

#include "test_mx_gemm_pipeline_ut_cases.inc"

TYPED_TEST(TEST_SUITE_NAME, MNPadding)
{
    if constexpr(TestFixture::PipelineType == MxGemmPipelineType::WeightPreshuffle ||
                 TestFixture::PipelineType == MxGemmPipelineType::CompEightWaves)
    {
        return;
    }

    std::vector<int> Ms{96, 160, 224};
    std::vector<int> Ns{96, 160, 224};
    std::vector<int> Ks;
    // K must be multiple of ScaleBlockSize (16 or 32) and K_Tile
    for(auto K_count : {2, 3, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int N : Ns)
        {
            for(int K : Ks)
            {
                this->template Run<true, true>(M, N, K);
            }
        }
    }
}

#undef TEST_SUITE_NAME
