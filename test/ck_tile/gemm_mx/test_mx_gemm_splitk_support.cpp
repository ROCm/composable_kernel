// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/epilogue/permuten_epilogue.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "ck_tile/host/device_prop.hpp"

namespace {

template <ck_tile::index_t VectorSize,
          template <typename...> class PipelineType = ck_tile::GemmPipelineAgBgCrCompAsync,
          template <typename...> class EpilogueType = ck_tile::CShuffleEpilogue>
struct BuildKernel
{
    using Row         = ck_tile::tensor_layout::gemm::RowMajor;
    using Col         = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Shape       = ck_tile::TileGemmShape<ck_tile::sequence<128, 128, 128>,
                                               ck_tile::sequence<2, 2, 1>,
                                               ck_tile::sequence<16, 16, 128>>;
    using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<Shape, 8, 4>;
    using Problem     = ck_tile::MxGemmPipelineProblem<
            ck_tile::fp8_t,
            ck_tile::fp8_t,
            float,
            Shape,
            ck_tile::TileGemmUniversalTraits<true, true, false, true, Row, Col, Row, true>,
            ck_tile::GemmPipelineScheduler::Intrawave>;
    using Pipeline        = PipelineType<Problem>;
    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ck_tile::fp8_t,
                                                             ck_tile::fp8_t,
                                                             ck_tile::tuple<>,
                                                             float,
                                                             ck_tile::fp16_t,
                                                             ck_tile::tuple<>,
                                                             Row,
                                                             ck_tile::element_wise::PassThrough,
                                                             128,
                                                             128,
                                                             2,
                                                             2,
                                                             16,
                                                             16,
                                                             128,
                                                             true,
                                                             1,
                                                             true,
                                                             VectorSize,
                                                             1,
                                                             false,
                                                             ck_tile::fp8_t,
                                                             ck_tile::fp8_t,
                                                             true>;
    using type = ck_tile::MxGemmKernel<Partitioner, Pipeline, EpilogueType<EpilogueProblem>>;
};

template <ck_tile::index_t VectorSize,
          template <typename...> class PipelineType = ck_tile::GemmPipelineAgBgCrCompAsync,
          template <typename...> class EpilogueType = ck_tile::CShuffleEpilogue>
bool supports_split_k(ck_tile::index_t split_k)
{
    using Kernel = typename BuildKernel<VectorSize, PipelineType, EpilogueType>::type;
    // No launch is needed: these dimensions isolate the epilogue's split-K support.
    const ck_tile::MxGemmHostArgs<> args({nullptr},
                                         {nullptr},
                                         {nullptr},
                                         {nullptr},
                                         {},
                                         nullptr,
                                         split_k,
                                         128,
                                         128,
                                         256,
                                         {256},
                                         {256},
                                         {},
                                         128);
    return Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args));
}

TEST(MxGemmSplitKSupport, ScalarCShuffleSupportsOrdinaryGemm)
{
    EXPECT_TRUE(supports_split_k<1>(1));
    EXPECT_FALSE(supports_split_k<1>(2));
}

TEST(MxGemmSplitKSupport, PairedCShuffleRetainsSplitK)
{
    EXPECT_TRUE(supports_split_k<2>(1));
    EXPECT_TRUE(supports_split_k<2>(2));
}

TEST(MxGemmSplitKSupport, TdmV1RejectsSplitK)
{
    const auto supports =
        supports_split_k<1, ck_tile::GemmPipelineAgBgCrCompTDMV1, ck_tile::TdmEpilogue>;
    EXPECT_FALSE(supports(2));
    // Accepting an argument also requires a gfx125 device.
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "TDM pipelines require a gfx125 device";
    }
    EXPECT_TRUE(supports(1));
}

TEST(MxGemmSplitKSupport, TdmV2RejectsSplitK)
{
    const auto supports =
        supports_split_k<1, ck_tile::GemmPipelineAgBgCrCompTDMV2, ck_tile::TdmEpilogue>;
    EXPECT_FALSE(supports(2));
    // Accepting an argument also requires a gfx125 device.
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "TDM pipelines require a gfx125 device";
    }
    EXPECT_TRUE(supports(1));
}

// comp_tdm_v3 declares kSupportsTdmSplitK: with CShuffle each split adds into C atomically and
// its TDM loads are clipped to its K range, so it takes k_batch > 1.
TEST(MxGemmSplitKSupport, TdmV3CShuffleAcceptsSplitK)
{
    const auto supports =
        supports_split_k<2, ck_tile::GemmPipelineAgBgCrCompTDMV3, ck_tile::CShuffleEpilogue>;
    // Accepting an argument also requires a gfx125 device.
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "TDM pipelines require a gfx125 device";
    }
    EXPECT_TRUE(supports(1));
    EXPECT_TRUE(supports(2));
}

// The other TDM pipelines do not declare it and keep rejecting MX split-K, even with an epilogue
// that could add the splits.
TEST(MxGemmSplitKSupport, TdmV1V2CShuffleRejectSplitK)
{
    const auto v1 =
        supports_split_k<2, ck_tile::GemmPipelineAgBgCrCompTDMV1, ck_tile::CShuffleEpilogue>;
    const auto v2 =
        supports_split_k<2, ck_tile::GemmPipelineAgBgCrCompTDMV2, ck_tile::CShuffleEpilogue>;
    EXPECT_FALSE(v1(2));
    EXPECT_FALSE(v2(2));
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "TDM pipelines require a gfx125 device";
    }
    EXPECT_TRUE(v1(1));
    EXPECT_TRUE(v2(1));
}

TEST(MxGemmSplitKSupport, PairedPermuteNRetainsSplitK)
{
    const auto supports =
        supports_split_k<2, ck_tile::GemmPipelineAgBgCrCompAsync, ck_tile::PermuteNEpilogue>;
    EXPECT_TRUE(supports(1));
    EXPECT_TRUE(supports(2));
}

// Declares both split-K markers: kAtomicAddRequiresEvenVectorSize, inherited from
// CShuffleEpilogue, and SupportsSplitK = false. The second must win.
template <typename Problem>
struct OverwritingCShuffleEpilogue : ck_tile::CShuffleEpilogue<Problem>
{
    static constexpr bool SupportsSplitK = false;
};

TEST(MxGemmSplitKSupport, OverwritingEpilogueRejectsSplitKDespiteAtomicAddMarker)
{
    using Kernel =
        BuildKernel<2, ck_tile::GemmPipelineAgBgCrCompAsync, OverwritingCShuffleEpilogue>::type;
    // The vector size is even, so the marker alone would allow split-K. The device dispatch
    // reads this constant, so false compiles the atomic_add arm out. The host assertions below
    // hold either way: UniversalGemmKernel's own k_batch guard also rejects k_batch = 2.
    EXPECT_FALSE(Kernel::kSplitKAtomicAddSupported);

    const auto supports =
        supports_split_k<2, ck_tile::GemmPipelineAgBgCrCompAsync, OverwritingCShuffleEpilogue>;
    EXPECT_TRUE(supports(1));
    EXPECT_FALSE(supports(2));
}

} // namespace
