// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Cases specific to GemmPipelineAgBgCrCompTDMRunAhead, shared by the default-ring and
// policy TUs. Include after defining TEST_SUITE_NAME.
#pragma once

#include <string>
#include <vector>

// Every step count from 1 to 2 * NumSlots + 1. That covers fewer steps than RunAhead, every
// remainder mod NumSlots, and the tail where retire's look-ahead shrinks from RunAhead - 1 to 0.
// Each runs with K whole, K ending half way into the last K tile, and K ending one element into
// it. One output tile, so this adds no kernel instantiation.
TYPED_TEST(TEST_SUITE_NAME, RingTailCoverage)
{
    using Defaults         = PipelineDefaultParams<TestFixture::PipelineType>;
    using Kernel           = typename TestFixture::template KernelFor<Defaults::PadM,
                                                                      Defaults::PadN,
                                                                      Defaults::PadK,
                                                                      Defaults::Preshuffle>::Kernel;
    constexpr int KT       = TestFixture::K_Tile;
    constexpr int NumSlots = Kernel::GemmPipeline::NumSlots;

    std::vector<int> num_loops;
    for(int nl = 1; nl <= 2 * NumSlots + 1; ++nl)
        num_loops.push_back(nl);

    for(int k_short : {0, KT / 2, KT - 1})
    {
        run_tail_coverage(
            KT,
            {{TestFixture::M_Tile, TestFixture::N_Tile}},
            k_short,
            [this](int M, int N, int K) { this->Run(M, N, K); },
            num_loops);
    }
}

// Host only. Without these checks a policy entry that silently fell back to <4, 3>, or a
// single-wave entry that still split the fills, would pass every GPU case above.
TYPED_TEST(TEST_SUITE_NAME, RingConfiguration)
{
    using Defaults  = PipelineDefaultParams<TestFixture::PipelineType>;
    using Kernel    = typename TestFixture::template KernelFor<Defaults::PadM,
                                                               Defaults::PadN,
                                                               Defaults::PadK,
                                                               Defaults::Preshuffle>::Kernel;
    using Pipeline  = typename Kernel::GemmPipeline;
    using TagPolicy = pipeline_tag_policy_t<std::tuple_element_t<13, TypeParam>>;
    using Expected  = std::conditional_t<std::is_void_v<TagPolicy>,
                                         ck_tile::GemmPipelineAgBgCrCompTDMRunAheadPolicy<>,
                                         TagPolicy>;

    EXPECT_EQ(Pipeline::NumSlots, Expected::NumSlots);
    EXPECT_EQ(Pipeline::RunAhead, Expected::RunAhead);

    const std::string name = Pipeline::GetName();
    const std::string suffix =
        "_" + std::to_string(Expected::NumSlots) + "x" + std::to_string(Expected::RunAhead);
    ASSERT_GE(name.size(), suffix.size()) << name;
    EXPECT_EQ(name.substr(name.size() - suffix.size()), suffix) << name;

    using Warps             = typename TestFixture::BlockWarps;
    constexpr int num_waves = Warps::at(ck_tile::number<0>{}) * Warps::at(ck_tile::number<1>{}) *
                              Warps::at(ck_tile::number<2>{});
    EXPECT_EQ(Pipeline::FillsPerStep, num_waves == 1 ? 2 : 1);

    // Every slot holds a whole A and B K tile; LDS padding can only add to that.
    using A = typename TestFixture::ADataType;
    using B = typename TestFixture::BDataType;
    const ck_tile::index_t slot_floor =
        TestFixture::M_Tile * TestFixture::K_Tile * static_cast<ck_tile::index_t>(sizeof(A)) +
        TestFixture::N_Tile * TestFixture::K_Tile * static_cast<ck_tile::index_t>(sizeof(B));
    EXPECT_EQ(Pipeline::GetSmemSize() % Expected::NumSlots, 0);
    EXPECT_GE(Pipeline::GetSmemSize(), Expected::NumSlots * slot_floor);
}

// TdmEpilogue overwrites C, so split-K must stay rejected on the host. Host only.
TYPED_TEST(TEST_SUITE_NAME, TdmEpilogueRejectsSplitK)
{
    constexpr int M = TestFixture::M_Tile;
    constexpr int N = TestFixture::N_Tile;
    constexpr int K = 4 * TestFixture::K_Tile;
    const auto stride_A =
        ck_tile::get_default_stride(M, K, 0, is_row_major(typename TestFixture::ALayout{}));
    const auto stride_B =
        ck_tile::get_default_stride(K, N, 0, is_row_major(typename TestFixture::BLayout{}));
    const auto stride_C =
        ck_tile::get_default_stride(M, N, 0, is_row_major(typename TestFixture::CLayout{}));

    EXPECT_TRUE(TestFixture::IsArgSupported(M, N, K, stride_A, stride_B, stride_C, 1));
    for(int k_batch : {2, 4})
        EXPECT_FALSE(TestFixture::IsArgSupported(M, N, K, stride_A, stride_B, stride_C, k_batch))
            << "k_batch=" << k_batch;
}
