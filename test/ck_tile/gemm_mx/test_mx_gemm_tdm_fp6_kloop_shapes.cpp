// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief FP6 (E2M3) MX GEMM with bf16 C on the gfx1250 TDM pipelines over K loops of 1 to 48
 *        tiles, checked with the tolerance derived in test_mx_gemm_host_helpers.hpp.
 * @details M and N span only a few tiles so that even the long K loops run quickly; one case has a
 *          single row (M = 1). N is 5 tiles throughout, so none of these shapes is in
 *          test_mx_gemm_pipeline_smoke_cases.inc.
 */

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmTdmFp6KLoop
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmFp6KLoop<T>>
{
    public:
    static constexpr bool kUseDerivedMxTolerance = true;

    // Every row is chosen to run; an unsupported combination must not turn into a silent skip.
    static_assert(
        TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmFp6KLoop<T>>::check_data_type(),
        "every FP6 row must have WMMA traits");
};

TYPED_TEST_SUITE(TestCkTileMxGemmTdmFp6KLoop, KernelTypesMxGemmTdmFp6KLoop);

TYPED_TEST(TestCkTileMxGemmTdmFp6KLoop, SingleRowFourKTiles)
{
    this->Run(1, 5 * TestFixture::N_Tile, 4 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmFp6KLoop, OneKTile)
{
    this->Run(2 * TestFixture::M_Tile, 5 * TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmFp6KLoop, TwoKTiles)
{
    this->Run(2 * TestFixture::M_Tile, 5 * TestFixture::N_Tile, 2 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmFp6KLoop, FourKTiles)
{
    this->Run(2 * TestFixture::M_Tile, 5 * TestFixture::N_Tile, 4 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmFp6KLoop, SixteenKTiles)
{
    this->Run(2 * TestFixture::M_Tile, 5 * TestFixture::N_Tile, 16 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmFp6KLoop, TwentyKTiles)
{
    this->Run(3 * TestFixture::M_Tile, 5 * TestFixture::N_Tile, 20 * TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileMxGemmTdmFp6KLoop, FortyEightKTiles)
{
    this->Run(3 * TestFixture::M_Tile, 5 * TestFixture::N_Tile, 48 * TestFixture::K_Tile);
}
