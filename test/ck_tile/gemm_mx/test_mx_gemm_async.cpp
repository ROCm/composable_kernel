// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_config.hpp"
#include "test_mx_gemm_util.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
using F4  = ck_tile::pk_fp4_t;
using F8  = ck_tile::fp8_t;

// clang-format off
using MxTypes = ::testing::Types<std::tuple<F4, F4, MX_GemmConfig16,               Row, Col, Row>,
                                 std::tuple<F4, F4, MX_GemmConfigEightWaves,       Row, Col, Row>,
                                 std::tuple<F4, F4, MXfp4_GemmConfig16_Preshuffle, Row, Col, Row>,
                                 std::tuple<F4, F4, MXfp4_GemmConfig16_PermuteN,   Row, Col, Row>,
                                 std::tuple<F8, F8, MX_GemmConfig16,               Row, Col, Row>,
                                 std::tuple<F8, F8, MX_GemmConfigEightWaves,       Row, Col, Row>,
                                 std::tuple<F8, F8, MXfp8_GemmConfig16_Preshuffle, Row, Col, Row>,
                                 std::tuple<F8, F8, MXfp8_GemmConfig16_PermuteN,   Row, Col, Row>>;
// clang-format on

template <typename TypeParam>
class TestMxGemm : public TestMxGemmUtil<TypeParam>
{
};

TYPED_TEST_SUITE(TestMxGemm, MxTypes);

TYPED_TEST(TestMxGemm, Default)
{
    this->Run(128, 512, 256);
    this->Run(256, 512, 512);
    this->Run(1024, 1024, 1024);
}

// Preshuffle split-K coverage. MxTypes already exercises the preshuffle configs on the
// non-split-K shapes (TestMxGemm.Default); this fixture pins the split-K shapes to the
// fp4/fp8 preshuffle configs.
using MxTypesPreshuffle =
    ::testing::Types<std::tuple<F4, F4, MXfp4_GemmConfig16_Preshuffle, Row, Col, Row>,
                     std::tuple<F8, F8, MXfp8_GemmConfig16_Preshuffle, Row, Col, Row>>;

template <typename TypeParam>
class TestMxGemmPreshuffle : public TestMxGemmUtil<TypeParam>
{
};

TYPED_TEST_SUITE(TestMxGemmPreshuffle, MxTypesPreshuffle);

// Split-K for the preshuffle pipeline: each k_id offsets the flat-B window and the
// host-preshuffled A/B scale windows into its own K slice (and accumulates via atomic-add).
// K is a multiple of K_Tile * k_batch (= 256 * k_batch); N is a multiple of 512 so the shapes
// are valid for both the fp4 (N_Tile = 512) and fp8 (N_Tile = 256) preshuffle configs.
TYPED_TEST(TestMxGemmPreshuffle, SplitK)
{
    this->Run(128, 512, 512, /*k_batch=*/2);
    this->Run(128, 512, 1024, /*k_batch=*/2);
    this->Run(128, 512, 1024, /*k_batch=*/4);
    this->Run(256, 512, 2048, /*k_batch=*/4);
}

// Regression coverage for the MX GEMM correctness fixes (PR #6663): num_loop == 3 hot-loop
// dispatch, split-K, and M/N padding. Shapes are pinned to fp8 x MX_GemmConfig16 (M_Tile = 64,
// N_Tile = 128, K_Tile = 256, default comp-async pipeline) so the regressions hit the intended
// code path -- e.g. K = 768 gives num_loop = K / K_Tile = 3.
using MxFp8Cfg16Types = ::testing::Types<std::tuple<F8, F8, MX_GemmConfig16, Row, Col, Row>>;

using MxFp8PadMNTypes =
    ::testing::Types<std::tuple<F8, F8, MXfp8_GemmConfig16_PadMN, Row, Col, Row>>;

template <typename TypeParam>
class TestMxGemmFp8Regression : public TestMxGemmUtil<TypeParam>
{
};

TYPED_TEST_SUITE(TestMxGemmFp8Regression, MxFp8Cfg16Types);

// num_loop == 3 must not enter the hot loop: with K_Tile = 256, K = 768 gives num_loop = 3,
// which previously produced 5 gemm accumulations instead of 3 (deterministically wrong).
TYPED_TEST(TestMxGemmFp8Regression, HotLoopTailNumLoopThree)
{
    this->Run(64, 128, 768);
    this->Run(128, 256, 768);
    this->Run(256, 256, 768);
}

// Split-K: exercises both the full_k_read and partial_k_read paths of SplitKBatchOffset together
// with the per-split scale-window K offset and the atomic-add epilogue. K is a multiple of
// K_Tile * k_batch and of WarpTile_K * k_batch (= 128 * k_batch) so every split lands on a
// packed-scale boundary.
TYPED_TEST(TestMxGemmFp8Regression, SplitK)
{
    this->Run(128, 256, 512, /*k_batch=*/2);
    this->Run(128, 256, 1024, /*k_batch=*/2);
    this->Run(128, 256, 1024, /*k_batch=*/4);
    this->Run(256, 256, 2048, /*k_batch=*/4);
}

// fp4 split-K (non-preshuffle). Same MX_GemmConfig16 tile shape as the fp8 regression above, so
// the K alignment requirements are identical; this verifies the packed (BPackedSize = 2) A/B
// pointer K-offset works under split-K + atomic-add for fp4.
using MxF4Cfg16Types = ::testing::Types<std::tuple<F4, F4, MX_GemmConfig16, Row, Col, Row>>;

template <typename TypeParam>
class TestMxGemmFp4Regression : public TestMxGemmUtil<TypeParam>
{
};

TYPED_TEST_SUITE(TestMxGemmFp4Regression, MxF4Cfg16Types);

TYPED_TEST(TestMxGemmFp4Regression, SplitK)
{
    this->Run(128, 256, 512, /*k_batch=*/2);
    this->Run(128, 256, 1024, /*k_batch=*/2);
    this->Run(128, 256, 1024, /*k_batch=*/4);
    this->Run(256, 256, 2048, /*k_batch=*/4);
}

template <typename TypeParam>
class TestMxGemmFp8PadMN : public TestMxGemmUtil<TypeParam>
{
};

TYPED_TEST_SUITE(TestMxGemmFp8PadMN, MxFp8PadMNTypes);

// M/N padding (kPadM = kPadN = true). M_Tile = 64, N_Tile = 128. Each of M and N must be >= its
// block tile (the CShuffleEpilogue cannot safely run a single partial tile along either
// dimension); K stays aligned because the MX async pipeline does not support K padding.
TYPED_TEST(TestMxGemmFp8PadMN, MNPaddingAligned)
{
    // Sanity: padding enabled but already-aligned M, N must not regress the normal path.
    this->Run(64, 128, 256);
}

TYPED_TEST(TestMxGemmFp8PadMN, MPadding)
{
    // M has a full tile + partial trailing tile (N aligned to N_Tile = 128).
    this->Run(96, 128, 256);
    this->Run(160, 128, 256);
}

TYPED_TEST(TestMxGemmFp8PadMN, NPadding)
{
    // N has a full tile + partial trailing tile (M aligned to M_Tile = 64).
    this->Run(64, 160, 256);
    this->Run(64, 224, 256);
}

TYPED_TEST(TestMxGemmFp8PadMN, MNPadding)
{
    // Both M and N unaligned (full + partial trailing tiles).
    this->Run(96, 160, 256);
    this->Run(160, 224, 512);
}
