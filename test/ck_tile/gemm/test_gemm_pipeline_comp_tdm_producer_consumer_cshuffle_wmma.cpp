// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GemmPipelineAgBgCrCompTDMProducerConsumer paired with CShuffleEpilogue. TdmEpilogue always
// overwrites C, so this pairing is the only way the producer/consumer pipeline can split K:
// every split atomically adds its partial tile into C. The fixture runs k_batch 1, 2 and 4.
//
// The shared test_gemm_pipeline_ut_cases.inc is not used: it feeds every TDM pipeline ragged N
// extents. TdmEpilogue clips them, but with row-major C the CShuffleEpilogue store would wrap a
// partial N tile into the next row, so the kernel rejects them for this pairing
// (UniversalGemmKernel::IsSupportedArgument). Ragged M is supported and run here, alongside the
// rejection tests.

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma
    : public TestCkTileGemmPipelineWmmaBase<
          T,
          TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma<T>>
{
    public:
    using Base = TestCkTileGemmPipelineWmmaBase<
        T,
        TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma<T>>;

    // Host-side check with the default strides for an M x N x K problem.
    static bool Supported(int M, int N, int K, int k_batch)
    {
        const auto stride_A =
            ck_tile::get_default_stride(M, K, 0, is_row_major(typename Base::ALayout{}));
        const auto stride_B =
            ck_tile::get_default_stride(K, N, 0, is_row_major(typename Base::BLayout{}));
        const auto stride_C =
            ck_tile::get_default_stride(M, N, 0, is_row_major(typename Base::CLayout{}));
        return Base::IsArgSupported(M, N, K, stride_A, stride_B, stride_C, k_batch);
    }
};

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma,
                 KernelTypesCompTDMProducerConsumerCShuffleWmma);

TYPED_TEST(TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma, SingleTile)
{
    for(int k_tiles : {1, 2, 4, 8})
    {
        this->Run(TestFixture::M_Tile, TestFixture::N_Tile, k_tiles * TestFixture::K_Tile);
    }
}

TYPED_TEST(TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma, MultiTile)
{
    this->Run(4 * TestFixture::M_Tile, 3 * TestFixture::N_Tile, 16 * TestFixture::K_Tile);
}

// The regime split-K exists for: few output tiles, a long K walk per tile.
TYPED_TEST(TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma, DeepK)
{
    this->Run(TestFixture::M_Tile, 2 * TestFixture::N_Tile, 128 * TestFixture::K_Tile);
}

// A ragged last K tile is clipped by the TDM loads. Run() only splits K into whole K tiles, so
// these shapes run at k_batch 1 alone.
TYPED_TEST(TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma, PartialKTile)
{
    for(int K : {TestFixture::K_Tile / 2, 3 * TestFixture::K_Tile - TestFixture::K_Tile / 2})
    {
        this->Run(TestFixture::M_Tile, TestFixture::N_Tile, K);
    }
}

TYPED_TEST(TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma, SplitKArgumentSupport)
{
    constexpr int M  = TestFixture::M_Tile;
    constexpr int N  = TestFixture::N_Tile;
    constexpr int KT = TestFixture::K_Tile;

    // The runs above only split K if the fixture schedules k_batch > 1 for this pipeline type;
    // Run() drops non-dividing splits silently, so pin the schedule here.
    EXPECT_EQ(this->k_batches_, (std::vector<int>{1, 2, 4}));

    // If the epilogue could not accumulate (e.g. an odd half-precision C vector), every k_batch
    // > 1 would be rejected and the split-K runs above would silently cover nothing.
    EXPECT_TRUE(TestFixture::Supported(M, N, 4 * KT, 2));
    EXPECT_TRUE(TestFixture::Supported(M, N, 4 * KT, 4));

    // Each split must hold whole K tiles.
    EXPECT_FALSE(TestFixture::Supported(M, N, 3 * KT, 2));
    EXPECT_FALSE(TestFixture::Supported(M, N, 4 * KT + KT / 2, 2));
    EXPECT_TRUE(TestFixture::Supported(M, N, 4 * KT + KT / 2, 1));
    // Non-power-of-two split counts are fine as long as each split holds whole K tiles.
    EXPECT_TRUE(TestFixture::Supported(M, N, 6 * KT, 3));
    EXPECT_FALSE(TestFixture::Supported(M, N, 4 * KT, 3));
}

// A ragged last M tile: with row-major C the rows past M lie beyond C's buffer range, or are
// masked by the padded view for bf16 atomic_add, so they are never written.
TYPED_TEST(TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma, RaggedM)
{
    for(int M : {TestFixture::M_Tile / 2, TestFixture::M_Tile + TestFixture::M_Tile / 2})
    {
        this->Run(M, TestFixture::N_Tile, 8 * TestFixture::K_Tile);
    }
}

TYPED_TEST(TestCkTileGemmPipelineCompTDMProducerConsumerCShuffleWmma, RaggedContiguousDimRejected)
{
    static_assert(is_row_major(typename TestFixture::Base::CLayout{}),
                  "these expectations assume row-major C, whose contiguous dimension is N");
    constexpr int M = TestFixture::M_Tile;
    constexpr int N = TestFixture::N_Tile;
    constexpr int K = 4 * TestFixture::K_Tile;

    // TDM pipelines cannot pad, and a partial tile along C's contiguous dimension would wrap into
    // the next row of C. It is rejected at every k_batch; a partial M tile is not.
    for(int k_batch : {1, 2})
    {
        EXPECT_FALSE(TestFixture::Supported(M, N + 1, K, k_batch)) << "k_batch=" << k_batch;
        EXPECT_TRUE(TestFixture::Supported(M + 1, N, K, k_batch)) << "k_batch=" << k_batch;
        EXPECT_TRUE(TestFixture::Supported(2 * M, 2 * N, K, k_batch)) << "k_batch=" << k_batch;
    }
}
