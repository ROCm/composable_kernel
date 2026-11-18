// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Stream-K on the gfx1250 TDM pipelines (comp_tdm and comp_tdm_v2). The CShuffle epilogue serves
// every reduction strategy; the TDM epilogue overwrites C, so it pairs with Linear and Tree only.

#include <cstdint>

#include "test_gemm_streamk_tdm_util.hpp"

struct CompTDMV1
{
    template <typename Problem>
    using pipeline = ck_tile::GemmPipelineAgBgCrCompTDMV1<Problem>;
};

struct CompTDMV2
{
    template <typename Problem>
    using pipeline = ck_tile::GemmPipelineAgBgCrCompTDMV2<Problem>;
};

// Wave layouts (M x N x K waves); comp_tdm_v2 needs exactly four.
using Waves2x2 = ck_tile::sequence<2, 2, 1>;
using Waves4x2 = ck_tile::sequence<4, 2, 1>;
using Waves2x4 = ck_tile::sequence<2, 4, 1>;

template <typename Pipeline,
          typename Input,
          typename Reduction,
          typename Epilogue,
          typename Persistence,
          ck_tile::index_t Tile = 128,
          typename ALayout      = Row,
          typename BLayout      = Col,
          typename Waves        = Waves2x2>
using TdmStreamKParams = std::tuple<Pipeline,
                                    Input,
                                    Reduction,
                                    Epilogue,
                                    Persistence,
                                    ck_tile::number<Tile>,
                                    ALayout,
                                    BLayout,
                                    Waves>;

// comp_tdm on eight waves, with row-major A and column-major B.
template <typename Reduction, typename Epilogue, typename Persistence, typename Waves>
using TdmStreamKEightWaves =
    TdmStreamKParams<CompTDMV1, BF16, Reduction, Epilogue, Persistence, 128, Row, Col, Waves>;

#if GFX1250_STREAMK_SMOKE
// The gfx1250 emulator runs one workgroup at a time, so a Linear or Tree tile owner would wait
// forever for later workgroups. Cover both pipelines and persistence modes with atomic reduction.
using Gfx1250StreamKTdmTypes =
    ::testing::Types<TdmStreamKParams<CompTDMV1, BF16, Atomic, CShuffleStore, NonPersistent>,
                     TdmStreamKParams<CompTDMV2, BF16, Atomic, CShuffleStore, Persistent>,
                     TdmStreamKParams<CompTDMV2, F16, Atomic, CShuffleStore, NonPersistent>>;
#else
using Gfx1250StreamKTdmTypes = ::testing::Types<
    TdmStreamKParams<CompTDMV1, BF16, Atomic, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Linear, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Linear, TdmStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Tree, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Tree, TdmStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Atomic, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV1, BF16, Linear, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV1, BF16, Linear, TdmStore, Persistent>,
    TdmStreamKParams<CompTDMV1, BF16, Tree, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV1, BF16, Tree, TdmStore, Persistent>,
    TdmStreamKParams<CompTDMV2, BF16, Atomic, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, TdmStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, TdmStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, BF16, Atomic, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, TdmStore, Persistent>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, TdmStore, Persistent>,
    TdmStreamKParams<CompTDMV2, F16, Atomic, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, F16, Linear, TdmStore, Persistent>,
    TdmStreamKParams<CompTDMV2, F16, Tree, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Linear, TdmStore, NonPersistent, 64>,
    TdmStreamKParams<CompTDMV2, BF16, Atomic, CShuffleStore, Persistent, 64>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, TdmStore, NonPersistent, 64>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, TdmStore, NonPersistent, 128, Col, Row>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, CShuffleStore, Persistent, 128, Col, Row>,
    TdmStreamKParams<CompTDMV1, BF16, Atomic, CShuffleStore, NonPersistent, 128, Row, Row>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, TdmStore, NonPersistent, 128, Row, Row>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, TdmStore, NonPersistent, 128, Col, Col>,
    TdmStreamKParams<CompTDMV1, BF16, Atomic, CShuffleStore, Persistent, 128, Col, Col>,
    TdmStreamKEightWaves<Linear, CShuffleStore, NonPersistent, Waves4x2>,
    TdmStreamKEightWaves<Tree, TdmStore, Persistent, Waves4x2>,
    TdmStreamKEightWaves<Atomic, CShuffleStore, Persistent, Waves2x4>,
    TdmStreamKEightWaves<Linear, TdmStore, NonPersistent, Waves2x4>>;
#endif

template <typename Params>
using TdmStreamKConfig = StreamKTdmConfig<std::tuple_element_t<0, Params>::template pipeline,
                                          std::tuple_element_t<6, Params>,
                                          std::tuple_element_t<7, Params>,
                                          Row,
                                          std::tuple_element_t<1, Params>,
                                          std::tuple_element_t<1, Params>,
                                          F32,
                                          std::tuple_element_t<1, Params>,
                                          ck_tile::sequence<std::tuple_element_t<5, Params>::value,
                                                            std::tuple_element_t<5, Params>::value,
                                                            64>,
                                          std::tuple_element_t<8, Params>,
                                          ck_tile::sequence<16, 16, 32>,
                                          std::tuple_element_t<2, Params>,
                                          std::tuple_element_t<4, Params>,
                                          std::tuple_element_t<3, Params>>;

template <typename Params>
class TestGfx1250StreamKTdm : public ::testing::Test,
                              public StreamKTdmHarness<TdmStreamKConfig<Params>>
{
    public:
    using Harness = StreamKTdmHarness<TdmStreamKConfig<Params>>;
    using Harness::Run; // ::testing::Test::Run is private
    static constexpr ck_tile::index_t Tile = std::tuple_element_t<5, Params>::value;
};

TYPED_TEST_SUITE(TestGfx1250StreamKTdm, Gfx1250StreamKTdmTypes);

TYPED_TEST(TestGfx1250StreamKTdm, DataParallelOnly)
{
    constexpr auto T = TestFixture::Tile;
    this->Run(2 * T, 2 * T, 512, 4, TestFixture::Split::None);
}

TYPED_TEST(TestGfx1250StreamKTdm, SplitKSegments)
{
    // Eight workgroups share one tile with 1 to 5 K iterations each: both tails of the double
    // LDS buffer, with and without the hot loop.
    constexpr auto T = TestFixture::Tile;
    for(int loops : {1, 2, 3, 4, 5})
        this->Run(T, T, 8 * 64 * loops, 8, TestFixture::Split::Partial);
}

TYPED_TEST(TestGfx1250StreamKTdm, SegmentsAcrossTiles)
{
    // Workgroups that finish one tile with a short (odd) segment and start the next tile in the
    // same launch: the next segment refills the LDS that the clipped prefetch targets.
    constexpr auto T = TestFixture::Tile;
    this->Run(T, 3 * T, 3 * 64, 4, TestFixture::Split::MultiTile);
    this->Run(T, 2 * T, 5 * 64, 4, TestFixture::Split::MultiTile);
}

TYPED_TEST(TestGfx1250StreamKTdm, MixedDataParallelAndStreamK)
{
    // 33 tiles on eight workgroups: data-parallel tiles, then Stream-K segments that span two
    // tiles. Persistent kernels also reuse LDS across consecutive data-parallel tiles.
    constexpr auto T = TestFixture::Tile;
    this->Run(T, 33 * T, 512, 8, TestFixture::Split::MultiTile);
}

TYPED_TEST(TestGfx1250StreamKTdm, RaggedK)
{
    // The last K tile of a segment is clipped by TDM.
    constexpr auto T = TestFixture::Tile;
    this->Run(T, 3 * T, 1000, 8, TestFixture::Split::Partial);
    this->Run(T, 3 * T, 544, 5, TestFixture::Split::Partial);
}

TYPED_TEST(TestGfx1250StreamKTdm, RaggedMN)
{
    constexpr auto T = TestFixture::Tile;
    // A ragged M tile lies past the last row of C, which both epilogues drop.
    this->Run(T + 8, 2 * T, 1024, 8);
    // N is the contiguous dimension of C: the TDM epilogue clips a ragged N tile, but CShuffle
    // would store it into the next row, so IsSupportedArgument rejects it.
    this->Run(T, 2 * T + 8, 1024, 8, TestFixture::Split::Any, TestFixture::UseTdmEpilogue);
    this->Run(118, 102, 8 * 64 * 3, 8, TestFixture::Split::Any, TestFixture::UseTdmEpilogue);
}

#if !GFX1250_STREAMK_SMOKE
TYPED_TEST(TestGfx1250StreamKTdm, FullDevice)
{
    // The device's workgroup budget: a Linear or Tree workgroup waits for later ones, so every
    // Stream-K workgroup the occupancy query allows must be resident at once.
    constexpr auto T = TestFixture::Tile;
    const auto kargs = TestFixture::MakeKargs(4 * T, 4 * T, 64 * 255, 0);
    EXPECT_EQ(kargs.tile_partitioner.get_sk_ctas(), kargs.tile_partitioner.get_max_active_wgs());
    this->Run(4 * T, 4 * T, 64 * 255, 0, TestFixture::Split::Partial);
}
#endif

TYPED_TEST(TestGfx1250StreamKTdm, RejectsMisalignedPointers)
{
    // TDM addresses A, B and C in whole elements.
    constexpr auto T      = TestFixture::Tile;
    const auto misaligned = reinterpret_cast<void*>(std::uintptr_t{1});
    EXPECT_TRUE(TestFixture::Kernel::IsSupportedArgument(TestFixture::MakeKargs(T, T, 512, 8)));
    EXPECT_FALSE(TestFixture::Kernel::IsSupportedArgument(
        TestFixture::MakeKargs(T, T, 512, 8, misaligned, nullptr, nullptr)));
    EXPECT_FALSE(TestFixture::Kernel::IsSupportedArgument(
        TestFixture::MakeKargs(T, T, 512, 8, nullptr, nullptr, misaligned)));
}
