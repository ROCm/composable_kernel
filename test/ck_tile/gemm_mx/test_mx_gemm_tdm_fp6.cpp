// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief FP6 (E2M3) MX GEMM on the gfx1250 TDM pipelines, checked with the tolerance derived in
 *        test_mx_gemm_host_helpers.hpp and sized for the smoke stage.
 * @details The pipelines support FP6 operands in one configuration: row-major A, column-major B,
 *          a 16x16x128 warp tile, 32-element E8M0 scales, no padding and no data-cache prefetch.
 *          TestCkTileMxGemmTdmFp6 runs the smoke cases at a 128-deep K tile with fp16 and bf16 C
 *          on V1 and V2, plus one case with padded A and B rows. TestCkTileMxGemmTdmFp6DeepKTile
 *          checks K tiles of 256 and 384 with a few cases, and TestCkTileMxGemmTdmFp6TileLayouts
 *          other block tiles and warp layouts.
 */

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmTdmFp6 : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmFp6<T>>
{
    public:
    static constexpr bool kUseDerivedMxTolerance = true;

    // Every row is chosen to run; an unsupported combination must not turn into a silent skip.
    static_assert(TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmFp6<T>>::check_data_type(),
                  "every FP6 row must have WMMA traits");
};

#define TEST_SUITE_NAME TestCkTileMxGemmTdmFp6

TYPED_TEST_SUITE(TestCkTileMxGemmTdmFp6, KernelTypesMxGemmTdmFp6);

#include "test_mx_gemm_pipeline_smoke_cases.inc"

#undef TEST_SUITE_NAME

TYPED_TEST(TestCkTileMxGemmTdmFp6, PaddedLeadingDimensions)
{
    // A and B rows longer than K, by different amounts: the row pitch of the 12-byte packs must
    // come from each operand's own stride, not from K. The pads are one and three packs, so the
    // row pitches (0.75 K + 12 and 0.75 K + 36 bytes) are multiples of 4 bytes but not of 8 or 16.
    constexpr int K = 2 * TestFixture::K_Tile;
    this->Run(2 * TestFixture::M_Tile, 2 * TestFixture::N_Tile, K, K + 16, K + 48);
}

template <typename T>
class TestCkTileMxGemmTdmFp6DeepKTile
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmFp6DeepKTile<T>>
{
    public:
    static constexpr bool kUseDerivedMxTolerance = true;

    // Every row is chosen to run; an unsupported combination must not turn into a silent skip.
    static_assert(
        TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmFp6DeepKTile<T>>::check_data_type(),
        "every FP6 row must have WMMA traits");
};

TYPED_TEST_SUITE(TestCkTileMxGemmTdmFp6DeepKTile, KernelTypesMxGemmTdmFp6DeepKTile);

TYPED_TEST(TestCkTileMxGemmTdmFp6DeepKTile, OneKTile)
{
    // No hot loop with tail One.
    this->Run(2 * TestFixture::M_Tile, 2 * TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmFp6DeepKTile, TwoKTiles)
{
    // No hot loop with tail Two.
    this->Run(2 * TestFixture::M_Tile, 2 * TestFixture::N_Tile, 2 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmFp6DeepKTile, ThreeKTiles)
{
    // The hot loop with tail One.
    this->Run(2 * TestFixture::M_Tile, 3 * TestFixture::N_Tile, 3 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmFp6DeepKTile, FourKTiles)
{
    // The hot loop with tail Two, the path of every longer even K loop.
    this->Run(2 * TestFixture::M_Tile, 2 * TestFixture::N_Tile, 4 * TestFixture::K_Tile);
}

template <typename T>
class TestCkTileMxGemmTdmFp6TileLayouts
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmFp6TileLayouts<T>>
{
    public:
    static constexpr bool kUseDerivedMxTolerance = true;

    // Every row is chosen to run; an unsupported combination must not turn into a silent skip.
    static_assert(
        TestCkTileMxGemmPipelineWmmaBase<T,
                                         TestCkTileMxGemmTdmFp6TileLayouts<T>>::check_data_type(),
        "every FP6 row must have WMMA traits");
};

TYPED_TEST_SUITE(TestCkTileMxGemmTdmFp6TileLayouts, KernelTypesMxGemmTdmFp6TileLayouts);

TYPED_TEST(TestCkTileMxGemmTdmFp6TileLayouts, FourKTiles)
{
    // 2 x 2 block tiles; the hot loop with tail Two.
    this->Run(2 * TestFixture::M_Tile, 2 * TestFixture::N_Tile, 4 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmFp6TileLayouts, PartialTilesThreeKTiles)
{
    // M and N end inside a block tile; the hot loop with tail One.
    this->Run(2 * TestFixture::M_Tile + TestFixture::M_Tile / 4,
              2 * TestFixture::N_Tile + TestFixture::N_Tile / 2,
              3 * TestFixture::K_Tile);
}
