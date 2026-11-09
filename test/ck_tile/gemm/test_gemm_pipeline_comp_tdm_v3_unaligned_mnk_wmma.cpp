// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_v3 gtest (bf16 and fp8), M, N and K that are not multiples of the tile (SMOKE_TEST):
// M, N or K unaligned, all three unaligned with odd strides, a shape below one tile, and M = 1 or
// N = 1.
//
// TDM clips every A/B box to the real extent (it handles ragged extents itself), so A/B never limit
// the shape. C is written by the CShuffle epilogue with vector stores: an unaligned N needs kPadN,
// and N must be a multiple of the C vector size, or a partial vector would write past the row end
// into the next row. For each shape the kernel must accept or reject it by that rule
// (ExpectSupported), and every shape it accepts must be correct:
// - the unpadded instances (V3Bf16Configs) accept an unaligned M or K and reject an unaligned N;
// - the padded instances (V3Bf16PaddedConfigs) also accept an unaligned N, for every N that
//   their C vector size divides (8 by default; 2 and 1 for the narrow-vector instances);
// - the TdmEpilogue instances (V3Bf16TdmEpilogueConfigs), unpadded, accept every shape: the TDM
//   store clips C in hardware.
// - early refill (RefillPoint::AfterArrive), which has its own K-tile bookkeeping: bf16 with
//   TdmEpilogue (V3Bf16AfterArriveConfigs) on every shape, and fp8 (V3Fp8UnalignedConfigs) with
//   the default policy and early refill, unpadded CShuffle as fp8 ships (an unaligned N rejected)
//   and early refill with TdmEpilogue.
// - split-K (k_batch 3) over the same shapes: the padded CShuffle instances with a split-K store
//   path (a C vector of at least 2) accept every shape their C store does whose K has at least 3
//   warp-tile K steps, and must be correct there; the C-vector-1 and TdmEpilogue instances reject
//   every k_batch > 1.
// Inputs and checks: test_gemm_pipeline_comp_tdm_v3_util.hpp.

#include "test_gemm_pipeline_comp_tdm_v3_util.hpp"

namespace {

struct Shape
{
    ck_tile::index_t M, N, K;
};

template <typename Config>
std::vector<Shape> unaligned_shapes()
{
    constexpr ck_tile::index_t TK = Config::TileK;
    return {
        {300, 256, 2 * TK},     // M unaligned
        {256, 300, 2 * TK},     // N unaligned, N % 8 = 4
        {256, 264, 2 * TK},     // N unaligned, N % 8 = 0
        {256, 256, 2 * TK + 8}, // K unaligned
        {256, 256, TK + 1},     // K unaligned and odd: odd A/B row strides
        {333, 517, 1000},       // M, N and K unaligned and odd
        {37, 45, 40},           // below one tile in M, N and K
        {37, 48, 40},           // below one tile, N % 8 = 0
        {1, 256, 2 * TK},       // M = 1
        {256, 1, 2 * TK},       // N = 1
    };
}

template <typename Config>
class TestCompTDMV3Bf16UnalignedMnk : public TestCkTileGemmPipelineCompTDMV3<Config>
{
    protected:
    void RunUnaligned(Init init, ck_tile::index_t k_batch = 1)
    {
        int ran = 0;
        for(const Shape& s : unaligned_shapes<Config>())
        {
            if(this->ExpectSupported(s.M, s.N, s.K, k_batch))
            {
                this->Run(s.M, s.N, s.K, init, k_batch);
                ++ran;
            }
        }
        // An instance without a split-K store path (TdmEpilogue, or a C vector of 1) accepts no
        // k_batch > 1; the others accept some of these shapes.
        EXPECT_EQ(ran > 0, k_batch == 1 || this->SplitKSupported(1 << 20, k_batch));
    }
};
template <typename Config>
class TestCompTDMV3Bf16UnalignedMnkTdmEpilogue : public TestCompTDMV3Bf16UnalignedMnk<Config>
{
};
template <typename Config>
class TestCompTDMV3Bf16UnalignedMnkAfterArrive : public TestCompTDMV3Bf16UnalignedMnk<Config>
{
};
template <typename Config>
class TestCompTDMV3Fp8UnalignedMnk : public TestCompTDMV3Bf16UnalignedMnk<Config>
{
};
template <typename Config>
class TestCompTDMV3Bf16UnalignedMnkPadded : public TestCompTDMV3Bf16UnalignedMnk<Config>
{
};

TYPED_TEST_SUITE(TestCompTDMV3Bf16UnalignedMnk, V3Bf16Configs);
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnk, IntUnaligned) { this->RunUnaligned(Init::Int); }
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnk, RandomUnaligned) { this->RunUnaligned(Init::Random); }

TYPED_TEST_SUITE(TestCompTDMV3Bf16UnalignedMnkPadded, V3Bf16PaddedConfigs);
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkPadded, IntUnaligned) { this->RunUnaligned(Init::Int); }
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkPadded, RandomUnaligned)
{
    this->RunUnaligned(Init::Random);
}
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkPadded, SparseUnalignedSplitK)
{
    this->RunUnaligned(Init::Sparse, 3);
}
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkPadded, RandomUnalignedSplitK)
{
    this->RunUnaligned(Init::Random, 3);
}

TYPED_TEST_SUITE(TestCompTDMV3Bf16UnalignedMnkTdmEpilogue, V3Bf16TdmEpilogueConfigs);
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkTdmEpilogue, IntUnaligned)
{
    this->RunUnaligned(Init::Int);
}
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkTdmEpilogue, RandomUnaligned)
{
    this->RunUnaligned(Init::Random);
}
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkTdmEpilogue, SplitKRejected)
{
    this->RunUnaligned(Init::Sparse, 3);
}

TYPED_TEST_SUITE(TestCompTDMV3Bf16UnalignedMnkAfterArrive, V3Bf16AfterArriveConfigs);
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkAfterArrive, IntUnaligned)
{
    this->RunUnaligned(Init::Int);
}
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkAfterArrive, RandomUnaligned)
{
    this->RunUnaligned(Init::Random);
}

TYPED_TEST_SUITE(TestCompTDMV3Fp8UnalignedMnk, V3Fp8UnalignedConfigs);
TYPED_TEST(TestCompTDMV3Fp8UnalignedMnk, IntUnaligned) { this->RunUnaligned(Init::Int); }
TYPED_TEST(TestCompTDMV3Fp8UnalignedMnk, RandomUnaligned) { this->RunUnaligned(Init::Random); }

} // namespace
