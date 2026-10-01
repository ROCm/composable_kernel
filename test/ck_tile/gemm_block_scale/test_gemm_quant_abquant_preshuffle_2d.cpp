// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// Type combinations for ABQuant tests
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantPreshuffleBTypes = ::testing::Types<
    // 1D B-scales; PreshuffleQuant = false && TransposeC = false (RCR layout with RowMajor AQ)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigPreshuffleBPrefill, GroupSize1D_128, GroupSize1D_128, ColumnMajor>, 
    /// 2D B-scales; PreshuffleQuant = false && TransposeC = true (RCR layout with RowMajor AQ)
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigPreshuffleBPrefillTransposeC, GroupSize1D_128, GroupSize2D128N, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigPreshuffleBPrefill, GroupSize1D_128, GroupSize2D128N, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigPreshuffleB_ABQuant_Prefill, GroupSize1D_128, GroupSize2D128N, ColumnMajor>
>;
// clang-format on

// Test suite for ABQuant
TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantPreshuffleBTypes);

// AQuant tests
TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

// Match the dispatcher's 2x2-warp prefill configuration. With TransposeC,
// a lane's C registers cover different N columns, each requiring its own BQ
// scale. The earlier transposed tests used group-N=128 and hid that mistake.
template <bool PreshuffleQuant, ck_tile::index_t TileK = 128>
struct GemmConfigABQuantTransposedPerColumn : public GemmConfigPreshuffleB_ABQuant_Prefill
{
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Tile = TileK;
    static constexpr bool BPreshuffleQuant   = PreshuffleQuant;
    static_assert(!TiledMMAPermuteN, "CShuffle requires ordinary B/BQ column order");
};

using GroupSize4N128K = ck_tile::QuantGroupShape<ck_tile::sequence<1, 4, 128>>;

// clang-format off
using ABQuantTransposedPerColumnTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigABQuantTransposedPerColumn<false>, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigABQuantTransposedPerColumn<true>, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigABQuantTransposedPerColumn<false>, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigABQuantTransposedPerColumn<true>, GroupSize1D_128, GroupSize1D_128, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigABQuantTransposedPerColumn<false, 256>, GroupSize1D_128, GroupSize4N128K, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigABQuantTransposedPerColumn<true, 256>, GroupSize1D_128, GroupSize4N128K, ColumnMajor>
>;
// clang-format on

template <typename Tuple>
class TestCkTileABQuantTransposedPerColumn : public TestCkTileGemmABQuant<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileABQuantTransposedPerColumn, ABQuantTransposedPerColumnTypes);

TYPED_TEST(TestCkTileABQuantTransposedPerColumn, NonuniformScales)
{
    // The fixture independently randomizes A, B, AQ and BQ and checks the
    // unmodified C result against reference_gemm_abquant. Include multiple
    // N tiles and different K-loop counts, including the saved failure shape.
    // The group-N=4 variants also exercise two K scales per block, where
    // preshuffled BQ carries the K-group in a lane instead of a register.
    constexpr auto k_ratio = TestFixture::GemmConfig::K_Tile / 128;
    this->run_test_with_validation(128, 128, 256 * k_ratio);
    this->run_test_with_validation(256, 128, 384 * k_ratio);
    this->run_test_with_validation(128, 256, 640 * k_ratio);
    this->run_test_with_validation(512, 256, 768 * k_ratio);
    this->run_test_with_validation(256, 384, 896 * k_ratio);
}
