// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief MX GEMM with K tails on the gfx1250 TDM pipelines (K not a multiple of the K tile, of 128
 *        or of the 32-element scale block), checked with the tolerance derived in
 *        test_mx_gemm_host_helpers.hpp.
 * @details TestCkTileMxGemmTdmKTail runs the FP8 and FP4 rows of KernelTypesMxGemmCompTDMWmma and
 *          KernelTypesMxGemmTdmBf16C that accept K tails (32-element scales, row-major A,
 *          column-major B, mixed pairs included) on TDM V1 and V2, plus rows with 256- and
 *          384-deep K tiles. Every case runs M = 144 and N = 112, so the last tiles in M and N are
 *          partial (for the 128-wide N tiles, N = 112 is less than one tile); apart from the
 *          whole-K-tile control cases, the last K tile is partial too. Every K up to one K tile,
 *          and every odd loop count, also makes the pipelines load a K tile entirely past K. The
 *          scale tensors have ceil(K / 32) columns, padded to a multiple of 4 (whole 4-scale
 *          packs); when K % 32 != 0 the last real column covers the remaining K % 32 elements.
 *          KTails runs the K values whose A and B rows are a multiple of 4 bytes, and
 *          NarrowPitchKTails those whose rows are an odd number of bytes or 2 bytes past a
 *          multiple of 4. On gfx1250 asicRevision 0 the harness runs only the non-cluster rows
 *          with 16x16 warp tiles and E8M0 scales (the bf16-C rows and the FP8 256- and 384-deep
 *          rows) and skips the others.
 *
 *          TestCkTileMxGemmTdmFp6KTail runs pk_fp6x16_t on both sides, in the one configuration
 *          the TDM pipelines support for FP6 (row-major A, column-major B, 16x16x128 warp tiles,
 *          32-element E8M0 scales): 128x128x128 block tiles on TDM V1 and V2 (fp16 and bf16 C,
 *          and bf16 C with cluster launch), and 64x64 block tiles with 256- and 384-deep K tiles.
 *          Its K values are multiples of 16, the elements of one 12-byte pack, so every A and B row
 *          is a multiple of 4 bytes. KTails runs M = 144 and N = 112 as above, and on the cluster
 *          rows also N = 240 (two N tiles); SingleRowKTail runs M = 1. On gfx1250 asicRevision 0
 *          the harness skips the cluster rows.
 */

#include <array>
#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_wmma_base.hpp"

// clang-format off
// Tuple format: ALayout, BLayout, CLayout, ADataType, BDataType, AScaleDataType, BScaleDataType,
// AccDataType, CDataType, M_Tile, N_Tile, K_Tile, M_Warp_Tile, N_Warp_Tile, PipelineType,
// ScaleBlockSize[, PermuteN, ClusterLaunch].
using KernelTypesMxGemmTdmKTail = ::testing::Types<
    // FP8 and FP4 rows of KernelTypesMxGemmCompTDMWmma with row-major A, column-major B and
    // 32-element scales: 64x64x128 block tiles, 32x32 warp tiles, fp16 C, TDM V1
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E5M3,  E5M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E4M3,  E4M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F4,    E8M0,  E5M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F8,    E5M3,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32>,
    // TDM V2: the V2 rows of the same list (no FP4 x FP4 E5M3 row; FP4 x FP8 with E4M3 A scales)
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E4M3,  E4M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F4,    E8M0,  E5M3,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F8,    E4M3,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32>,
    // cluster launch: 2 x 2 workgroups per cluster, each A and B tile multicast to the workgroups
    // that use it. The 3 x 2 output tiles round up to a 4 x 2 grid, so in the second cluster the
    // workgroups of M tile 3 exit at once while those of M tile 2 multicast B with a mask that
    // still includes them
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV1,  I32, std::false_type, ClusterEnable>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I128,       I32,        I32,        CompTDMV2,  I32, std::false_type, ClusterEnable>,
    // 16x16 warp tiles, bf16 C: the rows of KernelTypesMxGemmTdmBf16C
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I128,       I16,        I16,        CompTDMV2,  I32>,
    // 256- and 384-deep K tiles (2 and 3 scale packs). When ceil(K / 128) is not a multiple of the
    // packs per K tile, the last K tile reads scale packs past the ceil(K / 128) passed in (scales
    // of later rows, or zeros past the end of the scale view). Every K here exercises that read
    // except 160, 161, 240, 488 and 512 at 256 deep and 296, 768 and 1840 at 384 deep; the FP4
    // row does so with 32x32 warp tiles. A 384-deep K tile is three warp GEMMs deep, and K past
    // 256 within a K tile reaches the third.
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I256,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E5M3,  E5M3,      F32,       F16,        I64,         I64,          I256,       I32,        I32,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I384,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I384,       I16,        I16,        CompTDMV2,  I32>
>;
// clang-format on

namespace {

constexpr int kM = 144;
constexpr int kN = 112;

// A and B rows hold K contiguous elements: K bytes for FP8 and K / 2 bytes for FP4. All K values
// below are even, so that FP4 (two elements per byte) accepts them, and none is a multiple of 128.
// kAlignedPitchKs and kNarrowPitchKs together are kKTailKs of test_mx_gemm_tdm_gates.cpp, which
// checks that the gate accepts them; keep the lists in sync.

// K values whose rows are a multiple of 4 bytes for FP8 and FP4. Only 160 is a multiple of 32
// (K % 32 is 0, 8, 16 or 24), and a 128-deep K tile loops 1, 2, 3, 4 or 15 times.
constexpr std::array<int, 7> kAlignedPitchKs = {24, 80, 160, 240, 296, 488, 1840};

// K values whose rows are odd for FP4 K = 2 and 90 (1 and 45 bytes) and 2 bytes past a multiple of
// 4 for FP4 K = 92 (46 bytes) and FP8 K = 2 and 90; K % 32 is 2, 26 and 28.
constexpr std::array<int, 3> kNarrowPitchKs = {2, 90, 92};

// Odd K for the FP8 x FP8 configurations, whose A and B rows are then an odd number of bytes:
// K = 1, 33 (one past 32) and 161 (one past 128 + 32).
constexpr std::array<int, 3> kOddKs = {1, 33, 161};

} // namespace

template <typename T>
class TestCkTileMxGemmTdmKTail
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmKTail<T>>
{
    public:
    static constexpr bool kUseDerivedMxTolerance = true;

    // Every row is chosen to run; an unsupported combination must not turn into a silent skip.
    static_assert(
        TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmKTail<T>>::check_data_type(),
        "every K-tail row must be a supported MX type and scale combination with WMMA traits");

    // Runs M = kM, N = kN and the given K, with A and B rows of `stride` elements (0: K). Run
    // throws when IsSupportedArgument rejects the argument; report it and go on, so every K is
    // checked.
    void RunKTail(int k, int stride = 0)
    {
        EXPECT_NO_THROW(this->Run(kM, kN, k, stride, stride))
            << "K = " << k << ", stride = " << stride;
    }
};

TYPED_TEST_SUITE(TestCkTileMxGemmTdmKTail, KernelTypesMxGemmTdmKTail);

TYPED_TEST(TestCkTileMxGemmTdmKTail, KTails)
{
    static_assert(TestFixture::K_Tile == 128 || TestFixture::K_Tile == 256 ||
                      TestFixture::K_Tile == 384,
                  "add the whole-K-tile and K-tile-boundary cases for this K tile");

    for(int k : kAlignedPitchKs)
    {
        this->RunKTail(k);
    }

    // Rows wider than K: the elements between K and the 256-element stride are random, not zero.
    this->RunKTail(240, 256);

    if constexpr(TestFixture::K_Tile == 256)
    {
        // Whole K tiles, for comparison with the K tails of the same kernel.
        this->RunKTail(512);
        // Three K tiles, the last partial: the hot loop with an odd loop count, whose last K tile
        // reads a scale pack past the ceil(K / 128) = 5 passed in.
        this->RunKTail(600);
        // A multiple of 128 but not of the K tile: whole scale packs, the last K tile half full.
        this->RunKTail(640);
    }

    if constexpr(TestFixture::K_Tile == 384)
    {
        // Whole K tiles, for comparison with the K tails of the same kernel.
        this->RunKTail(768);
        // A multiple of 128 but not of the K tile: whole scale packs, the last K tile two warp
        // GEMMs deep.
        this->RunKTail(640);
        // Three full K tiles and a partial fourth.
        this->RunKTail(1200);
    }
}

TYPED_TEST(TestCkTileMxGemmTdmKTail, NarrowPitchKTails)
{
    using ADataType = typename TestFixture::ADataType;
    using BDataType = typename TestFixture::BDataType;

    for(int k : kNarrowPitchKs)
    {
        this->RunKTail(k);
    }

    if constexpr(std::is_same_v<ADataType, ck_tile::fp8_t> &&
                 std::is_same_v<BDataType, ck_tile::fp8_t>)
    {
        for(int k : kOddKs)
        {
            this->RunKTail(k);
        }
    }
}

// clang-format off
// FP6 x FP6 rows: row-major A, column-major B, 16x16x128 warp tiles and 32-element E8M0 scales, the
// one configuration the TDM pipelines support for FP6. Tuple format as above.
using KernelTypesMxGemmTdmFp6KTail = ::testing::Types<
    // 128x128x128 block tiles on 2 x 2 warps, TDM V1 and V2, fp16 and bf16 C
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I128,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,       I128,        I128,          I128,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,       F16,       I128,        I128,          I128,       I16,        I16,        CompTDMV2,  I32>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,       I128,        I128,          I128,       I16,        I16,        CompTDMV2,  I32>,
    // cluster launch: 2 x 2 workgroups per cluster, each A and B tile multicast to the workgroups
    // that use it. At M = 144 and N = 112 the 2 x 1 output tiles round up to a 2 x 2 grid: the
    // workgroups of N tile 1 exit at once, the two of N tile 0 share each B tile, and each
    // multicasts A with a mask that still includes the exited workgroup of its M tile (at M = 1,
    // B too). KTails also runs N = 240, two N tiles, where every A and B tile is shared by two
    // workgroups
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,       I128,        I128,          I128,       I16,        I16,        CompTDMV1,  I32, std::false_type, ClusterEnable>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,       I128,        I128,          I128,       I16,        I16,        CompTDMV2,  I32, std::false_type, ClusterEnable>,
    // 256- and 384-deep K tiles (2 and 3 scale packs), the rows of
    // KernelTypesMxGemmTdmFp6DeepKTile. When ceil(K / 128) is not a multiple of the packs per K
    // tile, the last K tile reads scale packs past the ceil(K / 128) passed in. Every K here
    // exercises that read except 144, 160, 240, 400 and 512 at 256 deep and 768 and 1840 at 384
    // deep.
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,       F16,        I64,         I64,          I256,       I16,        I16,        CompTDMV1,  I32>,
    std::tuple<    Row,     Col,     Row,       F6,        F6,    E8M0,  E8M0,      F32,      BF16,        I64,         I64,          I384,       I16,        I16,        CompTDMV2,  I32>
>;
// clang-format on

namespace {

// FP6 K values: each is a multiple of 16 (an FP6 row of K / 16 packs of 12 bytes, a multiple of 4
// bytes) and none of 128; K % 32 is 16 for all but 96 and 160, and 16 to 112 are below one 128-deep
// K tile. These are kF6KTailKs of test_mx_gemm_tdm_gates.cpp, which checks that the gate accepts
// them; keep the lists in sync.
constexpr std::array<int, 10> kF6KTailKs = {16, 48, 96, 112, 144, 160, 240, 400, 528, 1840};

// FP6 K values at M = 1: one 12-byte pack per row (the only K tile and the only scale block
// partial), and a partial last K tile and scale block after whole ones.
constexpr std::array<int, 2> kF6SingleRowKs = {16, 528};

// M of SingleRowKTail, and N of the cluster rows' case with two N tiles.
constexpr int kSingleRow = 1;
constexpr int kTwoNTiles = 240;

} // namespace

template <typename T>
class TestCkTileMxGemmTdmFp6KTail
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmFp6KTail<T>>
{
    public:
    using Base = TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmFp6KTail<T>>;

    static constexpr bool kUseDerivedMxTolerance = true;

    // Every row is chosen to run; an unsupported combination must not turn into a silent skip.
    static_assert(Base::check_data_type(),
                  "every FP6 K-tail row must be a supported MX type and scale combination with "
                  "WMMA traits");
    // Protects the rows of this suite: the K-tail gate takes FP6 only as pk_fp6x16_t on both
    // sides, and the K values and comments here assume its 16-element, 12-byte packs; FP8 and FP4
    // rows belong in TestCkTileMxGemmTdmKTail.
    static_assert(std::is_same_v<typename Base::ADataType, ck_tile::pk_fp6x16_t> &&
                      std::is_same_v<typename Base::BDataType, ck_tile::pk_fp6x16_t>,
                  "every FP6 K-tail row must have pk_fp6x16_t A and B, the only FP6 operands the "
                  "K-tail gate takes; FP8 and FP4 rows belong in TestCkTileMxGemmTdmKTail");

    // Runs M = m, N = n and the given K, with A and B rows of `stride` elements (0: K). Run throws
    // when IsSupportedArgument rejects the argument; report it and go on, so every K is checked.
    void RunKTail(int k, int stride = 0, int m = kM, int n = kN)
    {
        EXPECT_NO_THROW(this->Run(m, n, k, stride, stride))
            << "M = " << m << ", N = " << n << ", K = " << k << ", stride = " << stride;
    }
};

TYPED_TEST_SUITE(TestCkTileMxGemmTdmFp6KTail, KernelTypesMxGemmTdmFp6KTail);

TYPED_TEST(TestCkTileMxGemmTdmFp6KTail, KTails)
{
    static_assert(TestFixture::K_Tile == 128 || TestFixture::K_Tile == 256 ||
                      TestFixture::K_Tile == 384,
                  "add the whole-K-tile and K-tile-boundary cases for this K tile");

    for(int k : kF6KTailKs)
    {
        this->RunKTail(k);
    }

    // Rows wider than K: the elements between K and the 256-element stride are random, not zero.
    this->RunKTail(240, 256);

    if constexpr(TestFixture::K_Tile == 128)
    {
        // Whole K tiles, for comparison with the K tails of the same kernel: four K tiles, the
        // hot loop and an even loop count.
        this->RunKTail(512);
    }

    if constexpr(TestFixture::K_Tile == 256)
    {
        // Whole K tiles, for comparison with the K tails of the same kernel.
        this->RunKTail(512);
        // A multiple of 128 but not of the K tile: whole scale packs and three K tiles, the last
        // half full; that last K tile reads a scale pack past the ceil(K / 128) = 5 passed in.
        this->RunKTail(640);
    }

    if constexpr(TestFixture::K_Tile == 384)
    {
        // Whole K tiles, for comparison with the K tails of the same kernel.
        this->RunKTail(768);
        // A multiple of 128 but not of the K tile: whole scale packs, the last K tile two warp
        // GEMMs deep.
        this->RunKTail(640);
        // Three full K tiles and a partial fourth.
        this->RunKTail(1200);
    }

    if constexpr(TestFixture::ClusterLaunch)
    {
        // Two N tiles: the 2 x 2 output tiles fill the cluster, so each A tile, like each B tile,
        // is multicast to two workgroups that use it, through the hot loop and the K tile loaded
        // past K at the odd loop count (K = 528: five K tiles, the last 16 deep).
        this->RunKTail(528, 0, kM, kTwoNTiles);
    }
}

TYPED_TEST(TestCkTileMxGemmTdmFp6KTail, SingleRowKTail)
{
    // M = 1: one row of A, whose scales are padded to a whole M tile. At K = 16 that row is a
    // single 12-byte pack, so the only K tile and the only scale block are partial; at K = 528 a
    // partial last K tile and a partial last scale block follow whole ones. On the 128x128 rows
    // this is one output tile, so on the cluster rows both the A and the B multicast include
    // exited workgroups.
    for(int k : kF6SingleRowKs)
    {
        this->RunKTail(k, 0, kSingleRow);
    }
}
