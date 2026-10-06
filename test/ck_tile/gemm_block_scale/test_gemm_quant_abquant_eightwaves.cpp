// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
#ifdef CK_GFX950_SUPPORT
namespace {

template <ck_tile::index_t NGranularity>
struct EightWavesGranularityProblem
{
    using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, NGranularity, 128>>;

    struct BlockGemmShape
    {
        using WarpTile = ck_tile::sequence<16, 16, 128>;
    };
};

TEST(GemmABQuantEightWavesPolicy, RejectsBQuantGranularityFinerThanWarpNTile)
{
    using Policy = ck_tile::GemmABQuantPipelineAgBgCrAsyncPolicy;

    EXPECT_FALSE(Policy::IsBQuantNGranularitySupported<EightWavesGranularityProblem<1>>());
    EXPECT_TRUE(Policy::IsBQuantNGranularitySupported<EightWavesGranularityProblem<16>>());
    EXPECT_TRUE(Policy::IsBQuantNGranularitySupported<EightWavesGranularityProblem<128>>());
}

} // namespace

// Type combinations for ABQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantEightWavesTypes = ::testing::Types<
    // PreshuffleQuant = false && TransposeC = false (RCR layout with RowMajor AQ)
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigEightWaves, GroupSize1D_128, GroupSize2D128N, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigEightWaves_PreshuffleB, GroupSize1D_128, GroupSize2D128N, ColumnMajor>
>;
// clang-format on

// Test suite for ABQuant
TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantEightWavesTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
#endif
