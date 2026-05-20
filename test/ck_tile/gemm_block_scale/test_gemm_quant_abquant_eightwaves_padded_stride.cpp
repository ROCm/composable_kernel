// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for the EightWaves ABQuant pipeline on a B tensor whose
// leading-dim stride is larger than the packed value. The async B-load
// descriptor in the EightWaves policy must be built from the input view's
// real strides so that the kernel addresses B correctly when stride_B is
// larger than the inner length (e.g. row-aligned weight padding).

#include "test_gemm_quant_common.hpp"

using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
#ifdef CK_GFX950_SUPPORT
// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType, CDataType,
// QuantType, GemmConfig, AQuantGroupSize, BQuantGroupSize, BQLayout>
// clang-format off
using ABQuantEightWavesPaddedStrideTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, ColumnMajor, FP8, FP8, float, Half, ABQuantGrouped, GemmConfigEightWaves, GroupSize1D_128, GroupSize2D128N, ColumnMajor>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmABQuant, ABQuantEightWavesPaddedStrideTypes);

TYPED_TEST(TestCkTileGemmABQuant, ABQuantGroupedPaddedBStrideTest)
{
    // 256-byte row alignment for FP8 -> 256 elements of leading-dim padding.
    constexpr ck_tile::index_t k_batch      = 1;
    constexpr ck_tile::index_t stride_B_pad = 256;
    this->run_test_with_validation(1024, 1024, 1024, k_batch, stride_B_pad);
}
#endif
