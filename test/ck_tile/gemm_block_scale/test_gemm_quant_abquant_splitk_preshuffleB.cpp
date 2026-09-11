// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using GroupSize1x1x128   = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
using GroupSize1x128x128 = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// ABQuant split-K with B-preshuffle pipeline (WPABQuantBPipelineAgBgCrV2).
// Exercises both the regular (uniform-split) and the runtime-tail (uneven-split)
// dispatch paths, mirroring the non-preshuffle reject/accept tests in
// test_gemm_quant_abquant_splitk_decode.cpp.
//
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
//                QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>

// =====================================================================================
// Uniform-split tests: every split-K batch has the same per-batch num_loop and tail
// classification, so the regular (non-runtime-tail) dispatch path applies.
// =====================================================================================

// clang-format off
using ABQuantSplitKPreshuffleBUniformTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigPreshuffleBPrefill, GroupSize1x1x128, GroupSize1x128x128, ColumnMajor>,
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF8, BF8, float, Half, ABQuantGrouped, GemmConfigPreshuffleBPrefill, GroupSize1x1x128, GroupSize1x128x128, ColumnMajor>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantSplitKPreshuffleBUniformTypes);

// GemmConfigPreshuffleBPrefill: M_Tile=128, N_Tile=128, K_Tile=128, K_Warp_Tile=64.
// For uniform splits we want every batch to have the same num_loop classification:
// pick K such that KRead == KLast and num_loop_per_batch >= 2.

TYPED_TEST(TestCkTileGemmABQuant, PreshuffleB_SplitK2_K1024)
{
    // K=1024, k_batch=2 -> KRead=KLast=512, num_loop=4 per batch (Even tail).
    this->run_test_with_validation(128, 128, 1024, 2);
}

TYPED_TEST(TestCkTileGemmABQuant, PreshuffleB_SplitK4_K2048)
{
    // K=2048, k_batch=4 -> KRead=KLast=512, num_loop=4 per batch (Even tail).
    this->run_test_with_validation(128, 128, 2048, 4);
}

TYPED_TEST(TestCkTileGemmABQuant, PreshuffleB_SplitK2_LargeK_LargeN)
{
    // K=2048, larger N (multiple of N_Tile=128).
    this->run_test_with_validation(128, 256, 2048, 2);
}

// =====================================================================================
// Runtime-tail tests: K and k_batch chosen so the first split-K batch and the final
// (shorter) batch land in different (hot-loop, tail) classifications.  The default
// host-side fixed tail dispatch must reject this; the runtime-tail dispatch path
// (RuntimeSplitKTail=true) must accept it.  Uses the padded preshuffle config so
// uneven K passes the kPadK=false divisibility check (mirrors the non-preshuffle
// decode test against GemmConfigPadding).
// =====================================================================================

// clang-format off
using ABQuantSplitKPreshuffleBRuntimeTailTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigPreshuffleBPrefillPadded, GroupSize1x1x128, GroupSize1x128x128, ColumnMajor>
>;
// clang-format on

template <typename Tuple>
class TestCkTileGemmABQuantSplitKPreshuffleBReject : public TestCkTileGemmABQuant<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileGemmABQuantSplitKPreshuffleBReject,
                 ABQuantSplitKPreshuffleBRuntimeTailTypes);

// K=3328, k_batch=9 with K_Tile=128:
//   K_Warp_Tile is arch-dependent (gfx94/95: 64 for 8-bit, gfx12 WMMA: 16); for both
//   ceil(3328 / (9 * K_Warp_Tile)) * K_Warp_Tile = 384, so KRead is always a multiple of
//   BQuantGroupSize::kK = AQuantGroupSize::kK = 128 (Constraints 2/3 of IsSupportedArgument).
//   KLast = 3328 - 8*384 = 256.
//   num_loop_first = ceil(384/128) = 3  (hot_loop=true,  tail=Odd)
//   num_loop_last  = ceil(256/128) = 2  (hot_loop=false, tail=Even)
// Both hot_loop and tail differ, so the fixed host-side dispatch rejects; the runtime-tail
// dispatch path (RuntimeSplitKTail=true) accepts and dispatches per-batch.

TYPED_TEST(TestCkTileGemmABQuantSplitKPreshuffleBReject, RejectsMismatchedTailSplitK)
{
    EXPECT_THROW(this->run_test_with_validation(128, 128, 3328, 9), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmABQuantSplitKPreshuffleBReject, RuntimeTailAllowsMismatchedTailSplitK)
{
    this->run_test_with_validation(128, 128, 3328, 9, 0, true /* allow_runtime_splitk_tail */);
}
