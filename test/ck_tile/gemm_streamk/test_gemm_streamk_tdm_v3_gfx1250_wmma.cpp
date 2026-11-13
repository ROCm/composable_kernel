// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_tdm_util.hpp"
#include "gtest/gtest.h"

namespace {

// GemmPipelineAgBgCrCompTDMV3 supports 4-wave (2x2x1) and 16-wave (4x4x1) block
// configurations, with 16x16x32 WMMA warp tiles for 16-bit inputs and 32x32x128
// WMMA warp tiles for 8-bit inputs on gfx1250.
template <ck_tile::index_t MPerBlock,
          ck_tile::index_t NPerBlock,
          ck_tile::index_t KPerBlock,
          ck_tile::index_t WarpTileM   = 16,
          ck_tile::index_t WarpTileN   = 16,
          ck_tile::index_t WarpTileK   = 32,
          ck_tile::index_t BlockWarpsM = 2,
          ck_tile::index_t BlockWarpsN = 2>
struct GemmConfigTdmV3
{
    using BlockTile  = ck_tile::sequence<MPerBlock, NPerBlock, KPerBlock>;
    using BlockWarps = ck_tile::sequence<BlockWarpsM, BlockWarpsN, 1>;
    using WarpTile   = ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>;
};

// 4-wave (2x2x1) 16-bit tile configurations (16x16x32 warp tile)
using GemmConfigTdmV3_4W_64x64x64    = GemmConfigTdmV3<64, 64, 64>;
using GemmConfigTdmV3_4W_128x128x64  = GemmConfigTdmV3<128, 128, 64>;
using GemmConfigTdmV3_4W_128x128x128 = GemmConfigTdmV3<128, 128, 128>;

// 16-wave (4x4x1) 16-bit tile configuration (16x16x32 warp tile)
using GemmConfigTdmV3_16W_256x256x64 = GemmConfigTdmV3<256, 256, 64, 16, 16, 32, 4, 4>;

// 4-wave (2x2x1) 8-bit tile configurations (32x32x128 warp tile)
using GemmConfigTdmV3_4W_128x128x128_F8 = GemmConfigTdmV3<128, 128, 128, 32, 32, 128>;
using GemmConfigTdmV3_4W_128x128x256_F8 = GemmConfigTdmV3<128, 128, 256, 32, 32, 128>;

// 16-wave (4x4x1) 8-bit tile configurations (32x32x128 warp tile)
using GemmConfigTdmV3_16W_256x256x128_F8 = GemmConfigTdmV3<256, 256, 128, 32, 32, 128, 4, 4>;
using GemmConfigTdmV3_16W_256x256x256_F8 = GemmConfigTdmV3<256, 256, 256, 32, 32, 128, 4, 4>;

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename GemmConfig,
          typename Reduction,
          typename Persistence,
          typename Epilogue = CShuffleStore,
          typename Padding  = Unpadded>
using StreamKTdmV3Config = StreamKTdmConfig<ck_tile::GemmPipelineAgBgCrCompTDMV3,
                                            ALayout,
                                            BLayout,
                                            CLayout,
                                            ADataType,
                                            BDataType,
                                            AccDataType,
                                            CDataType,
                                            typename GemmConfig::BlockTile,
                                            typename GemmConfig::BlockWarps,
                                            typename GemmConfig::WarpTile,
                                            Reduction,
                                            Persistence,
                                            Epilogue,
                                            Padding>;

template <typename Config>
void run_streamk_tdm_v3_case(ck_tile::index_t M,
                             ck_tile::index_t N,
                             ck_tile::index_t K,
                             int num_wgs,
                             StreamKTdmSplit split = StreamKTdmSplit::Any,
                             bool supported        = true)
{
    StreamKTdmHarness<Config>::Run(M, N, K, num_wgs, split, supported);
}

template <typename Config>
class TestCkTileStreamKTdmV3Gfx1250 : public ::testing::Test, public StreamKTdmHarness<Config>
{
    protected:
    using Harness = StreamKTdmHarness<Config>;
    using Harness::Run; // ::testing::Test::Run is private

    void RunSplitCases()
    {
        // Single output tile (TileM x TileN) with 4 K iterations and max_active_wgs = 2:
        // forces a 2-CTA Stream-K split (2 K-tiles per CTA) and exercises cross-workgroup
        // accumulation for every tile shape (64x64, 128x128, 256x256).
        this->Run(Harness::TileM, Harness::TileN, 4 * Harness::TileK, 2, Harness::Split::Partial);
#if !GFX1250_STREAMK_SMOKE
        // 6 output tiles (2 * TileM x 3 * TileN) with 4 K iterations per tile and
        // max_active_wgs = 4: all 6 tiles are Stream-K, 6 K iterations per CTA, so each CTA
        // finishes one tile and starts the next in the same launch.
        this->Run(2 * Harness::TileM,
                  3 * Harness::TileN,
                  4 * Harness::TileK,
                  4,
                  Harness::Split::MultiTile);
#endif
    }
};

// clang-format off
using StreamKTdmV3Configs = ::testing::Types<
#if GFX1250_STREAMK_SMOKE
    // Emulator-safe Atomic smoke coverage across 4-wave and 16-wave, 16-bit and 8-bit,
    // all 4 A/B layouts, Persistent/NonPersistent, and Padded/Unpadded.
    // Note: TdmEpilogue overwrites C and pairs with Linear/Tree only.
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64,      Atomic, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64,      Atomic, Persistent,    CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Col, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_64x64x64,        Atomic, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Col, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64,      Atomic, Persistent,    CShuffleStore,    Padded>,
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_16W_256x256x64,     Atomic, Persistent,    CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_4W_128x128x256_F8,  Atomic, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_4W_128x128x128_F8,  Atomic, Persistent,    CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_16W_256x256x256_F8, Atomic, Persistent,    CShuffleStore,    Unpadded>
#else
    // 4-wave (2x2x1) 64x64x64 16-bit across all 4 A/B layouts
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_64x64x64,   Atomic, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_64x64x64,   Linear, Persistent,    CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Col, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_64x64x64,   Tree,   NonPersistent, TdmStore,         Unpadded>,
    StreamKTdmV3Config<Col, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_64x64x64,   Linear, Persistent,    TdmStore,         Padded>,

    // 4-wave (2x2x1) 128x128x64 16-bit across all 4 A/B layouts, reductions, persistence, and epilogues
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Atomic, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Linear, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Tree,   Persistent,    CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Linear, Persistent,    TdmStore,         Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Atomic, Persistent,    CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Linear, NonPersistent, TdmStore,         Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Tree,   Persistent,    CShuffleStore,    Padded>,
    StreamKTdmV3Config<Col, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Atomic, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Col, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Linear, Persistent,    TdmStore,         Unpadded>,
    StreamKTdmV3Config<Col, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Tree,   NonPersistent, CShuffleStore,    Padded>,
    StreamKTdmV3Config<Col, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Atomic, Persistent,    CShuffleStore,    Padded>,
    StreamKTdmV3Config<Col, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Linear, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Col, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Tree,   Persistent,    TdmStore,         Unpadded>,

    // 4-wave (2x2x1) 128x128x128 16-bit across all 4 A/B layouts
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x128, Atomic, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x128, Linear, Persistent,    TdmStore,         Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x128, Tree,   NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Col, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x128, Linear, Persistent,    CShuffleStore,    Padded>,
    StreamKTdmV3Config<Col, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x128, Tree,   Persistent,    TdmStore,         Unpadded>,

    // 16-wave (4x4x1) 256x256x64 16-bit across all 4 A/B layouts
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_16W_256x256x64, Atomic, Persistent,    CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_16W_256x256x64, Linear, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_16W_256x256x64, Tree,   Persistent,    TdmStore,         Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_16W_256x256x64, Linear, Persistent,    CShuffleStore,    Padded>,
    StreamKTdmV3Config<Col, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_16W_256x256x64, Linear, NonPersistent, TdmStore,         Unpadded>,
    StreamKTdmV3Config<Col, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_16W_256x256x64, Tree,   Persistent,    CShuffleStore,    Unpadded>,

    // 4-wave (2x2x1) 8-bit (fp8/bf8 -> bf16, 32x32x128 warp tile, RCR)
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_4W_128x128x128_F8, Atomic, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_4W_128x128x128_F8, Linear, Persistent,    TdmStore,         Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_4W_128x128x128_F8, Linear, Persistent,    CShuffleStore,    Padded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_4W_128x128x128_F8, Tree,   NonPersistent, TdmStore,         Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_4W_128x128x256_F8, Atomic, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_4W_128x128x256_F8, Linear, Persistent,    TdmStore,         Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_4W_128x128x256_F8, Tree,   Persistent,    CShuffleStore,    Unpadded>,

    // 16-wave (4x4x1) 8-bit (fp8/bf8 -> bf16, 32x32x128 warp tile, RCR)
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_16W_256x256x128_F8, Atomic, NonPersistent, CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_16W_256x256x128_F8, Linear, Persistent,    TdmStore,         Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_16W_256x256x256_F8, Atomic, Persistent,    CShuffleStore,    Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_16W_256x256x256_F8, Linear, NonPersistent, TdmStore,         Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_16W_256x256x256_F8, Tree,   Persistent,    CShuffleStore,    Unpadded>
#endif
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileStreamKTdmV3Gfx1250, StreamKTdmV3Configs);

TYPED_TEST(TestCkTileStreamKTdmV3Gfx1250, SplitAndMixedTileCases) { this->RunSplitCases(); }

// Three output tiles with one K iteration per tile on two workgroups puts all three tiles in
// Stream-K (sk_tiles == 3, sk_ctas == 2) with zero partial-tile segments, covering the full-tile
// path in StreamKKernel::StreamKGemm (store directly to C). Safe for GFX1250_STREAMK_SMOKE because
// no tile needs a cross-workgroup wait.
TEST(StreamKTdmV3EdgeCases, OneCtaFullTile)
{
    using Config4W  = StreamKTdmV3Config<Row,
                                         Col,
                                         Row,
                                         BF16,
                                         BF16,
                                         F32,
                                         BF16,
                                         GemmConfigTdmV3_4W_128x128x64,
                                         Linear,
                                         NonPersistent,
                                         CShuffleStore,
                                         Unpadded>;
    using Config16W = StreamKTdmV3Config<Row,
                                         Col,
                                         Row,
                                         BF16,
                                         BF16,
                                         F32,
                                         BF16,
                                         GemmConfigTdmV3_16W_256x256x64,
                                         Linear,
                                         Persistent,
                                         TdmStore,
                                         Unpadded>;

    run_streamk_tdm_v3_case<Config4W>(128, 3 * 128, 64, 2, StreamKTdmSplit::FullTile);
    run_streamk_tdm_v3_case<Config16W>(256, 3 * 256, 64, 2, StreamKTdmSplit::FullTile);
}

#if !GFX1250_STREAMK_SMOKE
// Exercises all four CompTDMV3 K-loop dispatch paths per CTA in a 2-CTA split:
//   - num_loop == 1 (has_hot_loop = false, TailNumber::One):  K = 2 * 64 = 128
//   - num_loop == 2 (has_hot_loop = false, TailNumber::Two):  K = 4 * 64 = 256
//   - num_loop == 3 (has_hot_loop = true,  TailNumber::Odd):  K = 6 * 64 = 384
//   - num_loop == 4 (has_hot_loop = true,  TailNumber::Even): K = 8 * 64 = 512
TEST(StreamKTdmV3EdgeCases, TailNumLoopPaths)
{
    using Config = StreamKTdmV3Config<Row,
                                      Col,
                                      Row,
                                      BF16,
                                      BF16,
                                      F32,
                                      BF16,
                                      GemmConfigTdmV3_4W_128x128x64,
                                      Linear,
                                      Persistent,
                                      CShuffleStore,
                                      Unpadded>;

    run_streamk_tdm_v3_case<Config>(128, 128, 128, 2, StreamKTdmSplit::Partial);
    run_streamk_tdm_v3_case<Config>(128, 128, 256, 2, StreamKTdmSplit::Partial);
    run_streamk_tdm_v3_case<Config>(128, 128, 384, 2, StreamKTdmSplit::Partial);
    run_streamk_tdm_v3_case<Config>(128, 128, 512, 2, StreamKTdmSplit::Partial);
}

// Unaligned M, N, and K with PadM=true, PadN=true, PadK=true
// under 1-CTA, 2-CTA, and multi-CTA Stream-K splits across all four A/B layouts,
// 4-wave and 16-wave configs, 16-bit and 8-bit inputs, and CShuffle/TDM epilogues.
TEST(StreamKTdmV3EdgeCases, PaddedUnalignedMNKAllLayouts)
{
    using RcrBf16_4W  = StreamKTdmV3Config<Row,
                                           Col,
                                           Row,
                                           BF16,
                                           BF16,
                                           F32,
                                           BF16,
                                           GemmConfigTdmV3_4W_128x128x64,
                                           Linear,
                                           Persistent,
                                           CShuffleStore,
                                           Padded>;
    using RrrFp16_4W  = StreamKTdmV3Config<Row,
                                           Row,
                                           Row,
                                           F16,
                                           F16,
                                           F32,
                                           F16,
                                           GemmConfigTdmV3_4W_128x128x64,
                                           Atomic,
                                           NonPersistent,
                                           CShuffleStore,
                                           Padded>;
    using CrrBf16_4W  = StreamKTdmV3Config<Col,
                                           Row,
                                           Row,
                                           BF16,
                                           BF16,
                                           F32,
                                           BF16,
                                           GemmConfigTdmV3_4W_64x64x64,
                                           Tree,
                                           Persistent,
                                           TdmStore,
                                           Padded>;
    using CcrFp16_4W  = StreamKTdmV3Config<Col,
                                           Col,
                                           Row,
                                           F16,
                                           F16,
                                           F32,
                                           F16,
                                           GemmConfigTdmV3_4W_128x128x64,
                                           Linear,
                                           NonPersistent,
                                           TdmStore,
                                           Padded>;
    using RcrBf16_16W = StreamKTdmV3Config<Row,
                                           Col,
                                           Row,
                                           BF16,
                                           BF16,
                                           F32,
                                           BF16,
                                           GemmConfigTdmV3_16W_256x256x64,
                                           Linear,
                                           Persistent,
                                           CShuffleStore,
                                           Padded>;
    using RcrF8_4W    = StreamKTdmV3Config<Row,
                                           Col,
                                           Row,
                                           F8,
                                           F8,
                                           F32,
                                           BF16,
                                           GemmConfigTdmV3_4W_128x128x128_F8,
                                           Linear,
                                           Persistent,
                                           CShuffleStore,
                                           Padded>;

    // 1-CTA, 2-CTA, and multi-CTA unaligned M/N/K across all four A/B layouts
    run_streamk_tdm_v3_case<RcrBf16_4W>(100, 120, 150, 1);
    run_streamk_tdm_v3_case<RcrBf16_4W>(100, 120, 200, 2);
    run_streamk_tdm_v3_case<RcrBf16_4W>(65, 72, 160, 2);
    run_streamk_tdm_v3_case<RcrBf16_4W>(193, 264, 225, 4);

    run_streamk_tdm_v3_case<RrrFp16_4W>(100, 120, 200, 2);
    run_streamk_tdm_v3_case<RrrFp16_4W>(200, 304, 200, 4);

    run_streamk_tdm_v3_case<CrrBf16_4W>(65, 72, 160, 2);
    run_streamk_tdm_v3_case<CrrBf16_4W>(150, 200, 180, 4);

    run_streamk_tdm_v3_case<CcrFp16_4W>(100, 120, 200, 2);
    run_streamk_tdm_v3_case<CcrFp16_4W>(200, 304, 200, 4);

    // 16-wave 256x256x64 padded unaligned M/N/K (2-CTA and multi-CTA)
    run_streamk_tdm_v3_case<RcrBf16_16W>(200, 240, 200, 2);
    run_streamk_tdm_v3_case<RcrBf16_16W>(400, 520, 200, 4);

    // 8-bit 128x128x128 padded unaligned M/N/K (2-CTA and multi-CTA)
    run_streamk_tdm_v3_case<RcrF8_4W>(100, 120, 300, 2);
    run_streamk_tdm_v3_case<RcrF8_4W>(200, 304, 300, 4);
}

// TDM clips A and B in hardware, so unaligned M and K work even when unpadded.
// For row-major C:
//   - Unpadded CShuffleEpilogue requires N % TileN == 0.
//   - Padded CShuffleEpilogue requires N % GetVectorSizeC() == 0 (8 elements for bf16/fp16).
//   - TdmEpilogue clips C in hardware and accepts any N, including N % 8 != 0.
TEST(StreamKTdmV3EdgeCases, UnpaddedAndOddNAlignment)
{
    using UnpaddedConfig = StreamKTdmV3Config<Row,
                                              Col,
                                              Row,
                                              BF16,
                                              BF16,
                                              F32,
                                              BF16,
                                              GemmConfigTdmV3_4W_128x128x64,
                                              Linear,
                                              NonPersistent,
                                              CShuffleStore,
                                              Unpadded>;
    using PaddedConfig   = StreamKTdmV3Config<Row,
                                              Col,
                                              Row,
                                              BF16,
                                              BF16,
                                              F32,
                                              BF16,
                                              GemmConfigTdmV3_4W_128x128x64,
                                              Linear,
                                              NonPersistent,
                                              CShuffleStore,
                                              Padded>;

    using UnpaddedTdmConfig    = StreamKTdmV3Config<Row,
                                                    Col,
                                                    Row,
                                                    BF16,
                                                    BF16,
                                                    F32,
                                                    BF16,
                                                    GemmConfigTdmV3_4W_128x128x64,
                                                    Linear,
                                                    NonPersistent,
                                                    TdmStore,
                                                    Unpadded>;
    constexpr auto Any         = StreamKTdmSplit::Any;
    constexpr bool Unsupported = false;

    // Unpadded CShuffleEpilogue: unaligned M and K with tile-aligned N succeeds.
    run_streamk_tdm_v3_case<UnpaddedConfig>(100, 128, 200, 2);
    // Unpadded CShuffleEpilogue: N not a multiple of TileN is rejected.
    run_streamk_tdm_v3_case<UnpaddedConfig>(128, 120, 256, 2, Any, Unsupported);

    // Padded CShuffleEpilogue: N = 120 (multiple of 8) succeeds; N = 102 (not multiple of 8)
    // is rejected.
    run_streamk_tdm_v3_case<PaddedConfig>(100, 120, 200, 2);
    run_streamk_tdm_v3_case<PaddedConfig>(100, 102, 200, 2, Any, Unsupported);

    // TdmEpilogue clips C in hardware, so N = 102 (not divisible by 8) succeeds in both
    // 2-CTA and multi-CTA splits.
    run_streamk_tdm_v3_case<UnpaddedTdmConfig>(100, 102, 200, 2);
    run_streamk_tdm_v3_case<UnpaddedTdmConfig>(200, 302, 200, 4);
}
#endif

} // namespace
