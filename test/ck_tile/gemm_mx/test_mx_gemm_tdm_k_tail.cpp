// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief MX GEMM with K tails on the gfx1250 TDM pipelines (K not a multiple of the K tile, of 128
 *        or of the 32-element scale block), checked with the tolerance derived in
 *        test_mx_gemm_host_helpers.hpp.
 * @details The rows are the FP8 and FP4 rows of KernelTypesMxGemmCompTDMWmma and
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
