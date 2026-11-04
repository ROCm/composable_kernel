// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Host-only checks of the MX GEMM argument gates and LDS sizing for the gfx125 TDM
 *        pipelines.
 * @details Nothing is launched. The shape and pointer-alignment checks, the policies' LDS padding,
 *          the LDS sizes of policies, pipelines and kernels, and the logged rejection reasons are
 *          checked on any host; that IsSupportedArgument accepts an argument also needs a gfx125
 *          device and is skipped without one.
 */

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace {

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using F8    = ck_tile::fp8_t;
using BF8   = ck_tile::bf8_t;
using F4    = ck_tile::pk_fp4_t;
using F6    = ck_tile::pk_fp6x16_t;
using BF6   = ck_tile::pk_bf6x16_t;
using F6x32 = ck_tile::pk_fp6x32_t;
using E8M0  = ck_tile::e8m0_t;
using E5M3  = ck_tile::e5m3_t;

using ck_tile::index_t;

// One packed int32 of TDM scales covers KXdlPack scale blocks (KXdlPackEff of the TDM pipelines).
constexpr index_t kTdmKXdlPack = 4;

template <typename A,
          typename B,
          typename ALayout       = Row,
          typename BLayout       = Col,
          index_t KPerBlock      = 128,
          index_t ScaleBlockSize = 32>
bool shape_ok(index_t M, index_t N, index_t K, index_t stride_A, index_t stride_B)
{
    return ck_tile::
        IsMxTdmShapeSupported<A, B, ALayout, BLayout, KPerBlock, ScaleBlockSize, kTdmKXdlPack>(
            M, N, K, stride_A, stride_B);
}

// Calls f() with CK_TILE_LOGGING on and returns its result together with what it logged.
template <typename F>
std::pair<bool, std::string> logged(F&& f)
{
    ck_tile::UpdateEnvVar(CK_TILE_ENV(CK_TILE_LOGGING), true);
    testing::internal::CaptureStderr();
    const bool result     = f();
    const std::string log = testing::internal::GetCapturedStderr();
    ck_tile::EnvUnset(CK_TILE_ENV(CK_TILE_LOGGING));
    return {result, log};
}

// The pipeline problem of TdmMxGemm: tile 64 x 64 x KTile, warps 2x2x1, a WarpTile x WarpTile x
// 128 warp tile and 32-element scales, as in the TDM gtests. It names no pipeline, so a policy can
// be checked with it on its own.
template <typename A,
          typename B,
          typename ALayout = Row,
          typename BLayout = Col,
          typename AScale  = E8M0,
          typename BScale  = E8M0,
          index_t KTile    = 128,
          index_t WarpTile = 16>
struct TdmMxProblem
{
    using Shape = ck_tile::TileGemmShape<ck_tile::sequence<64, 64, KTile>,
                                         ck_tile::sequence<2, 2, 1>,
                                         ck_tile::sequence<WarpTile, WarpTile, 128>>;
    using Traits =
        ck_tile::TileGemmUniversalTraits<false, false, false, true, ALayout, BLayout, Row, true>;
    using type = ck_tile::MxGemmPipelineProblem<A,
                                                B,
                                                float,
                                                Shape,
                                                Traits,
                                                ck_tile::GemmPipelineScheduler::Intrawave,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::element_wise::PassThrough,
                                                A,
                                                B,
                                                AScale,
                                                BScale,
                                                32>;
};

// An MX GEMM with a TDM pipeline over TdmMxProblem.
template <template <typename...> class Pipeline,
          typename A,
          typename B,
          typename ALayout = Row,
          typename BLayout = Col,
          typename AScale  = E8M0,
          typename BScale  = E8M0,
          index_t KTile    = 128,
          index_t WarpTile = 16>
struct TdmMxGemm
{
    using ProblemDef      = TdmMxProblem<A, B, ALayout, BLayout, AScale, BScale, KTile, WarpTile>;
    using Shape           = typename ProblemDef::Shape;
    using Partitioner     = ck_tile::GemmSpatiallyLocalTilePartitioner<Shape, 8, 4>;
    using Problem         = typename ProblemDef::type;
    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<A,
                                                             B,
                                                             ck_tile::tuple<>,
                                                             float,
                                                             ck_tile::half_t,
                                                             ck_tile::tuple<>,
                                                             Row,
                                                             ck_tile::element_wise::PassThrough,
                                                             64,
                                                             64,
                                                             2,
                                                             2,
                                                             WarpTile,
                                                             WarpTile,
                                                             128,
                                                             true,
                                                             1,
                                                             false,
                                                             1,
                                                             1,
                                                             true,
                                                             A,
                                                             B,
                                                             true>;
    using Kernel          = ck_tile::
        MxGemmKernel<Partitioner, Pipeline<Problem>, ck_tile::TdmEpilogue<EpilogueProblem>>;
    static_assert(Kernel::KXdlPackEff == kTdmKXdlPack);

    static typename Kernel::KernelArgs
    make_args(index_t M, index_t N, index_t K, index_t stride_A, index_t stride_B)
    {
        const ck_tile::MxGemmHostArgs<> args({nullptr},
                                             {nullptr},
                                             {nullptr},
                                             {nullptr},
                                             {},
                                             nullptr,
                                             1,
                                             M,
                                             N,
                                             K,
                                             {stride_A},
                                             {stride_B},
                                             {},
                                             N);
        return Kernel::MakeKernelArgs(args);
    }
};

// The policy a pipeline template uses, read from its default template argument without
// instantiating the pipeline, so that only the policy is checked.
template <typename Pipeline>
struct PipelinePolicyOf;

template <template <typename, typename> class PipelineTemplate, typename Problem, typename Policy>
struct PipelinePolicyOf<PipelineTemplate<Problem, Policy>>
{
    using type = Policy;
};

template <template <typename...> class Pipeline, typename A, typename B>
using TdmPolicy = typename PipelinePolicyOf<Pipeline<typename TdmMxProblem<A, B>::type>>::type;

// The TDM policies never pad 6-bit LDS rows, in any compilation pass: this file's host pass and
// each device pass check it for the V1 and V2 policies, so the host's GetSmemSize agrees with the
// gfx125 device's. Only the Problem is built, so each policy is checked on its own.
// PolicyF6LdsRowsAreDense below shows the check is not vacuous: the base policy pads these rows.
template <typename Policy, typename Problem, bool IsA>
constexpr bool kTdmLdsPadded =
    Policy::template GetLdsPaddingConfig<Problem, IsA>()[ck_tile::number<0>{}];

using F6Problem  = typename TdmMxProblem<F6, F6>::type;
using F6PolicyV1 = TdmPolicy<ck_tile::GemmPipelineAgBgCrCompTDMV1, F6, F6>;
using F6PolicyV2 = TdmPolicy<ck_tile::GemmPipelineAgBgCrCompTDMV2, F6, F6>;
static_assert(!kTdmLdsPadded<F6PolicyV1, F6Problem, true> &&
                  !kTdmLdsPadded<F6PolicyV1, F6Problem, false>,
              "6-bit LDS rows must not be padded (V1 policy)");
static_assert(!kTdmLdsPadded<F6PolicyV2, F6Problem, true> &&
                  !kTdmLdsPadded<F6PolicyV2, F6Problem, false>,
              "6-bit LDS rows must not be padded (V2 policy)");

// GetLdsPaddingConfig as {padded, pad_amount, pad_interval}.
template <typename Config>
std::array<index_t, 3> lds_padding(const Config& config)
{
    return {static_cast<index_t>(config[ck_tile::number<0>{}]),
            static_cast<index_t>(config[ck_tile::number<1>{}]),
            static_cast<index_t>(config[ck_tile::number<2>{}])};
}

// A dense 6-bit LDS row of TdmMxProblem's K tile: KPerBlock / 16 packs of 12 bytes.
using F6Shape = typename TdmMxProblem<F6, F6>::Shape;
constexpr index_t kF6LdsRowBytes =
    F6Shape::kK / ck_tile::numeric_traits<F6>::PackedSize * static_cast<index_t>(sizeof(F6));
// One LDS buffer of that tile: MPerBlock rows of A and NPerBlock rows of B, unpadded.
constexpr index_t kF6DenseLdsBufferBytes = (F6Shape::kM + F6Shape::kN) * kF6LdsRowBytes;

// A 256-byte aligned address, as hipMalloc returns, standing in for a device pointer: the TDM
// argument checks only look at its value.
constexpr std::uintptr_t kDeviceAddress = 0x10000;
constexpr char kAlignmentReason[] =
    "requires A/B/E pointers aligned to TDMElementTraits<T>::alignment";

// UniversalGemmKernel::IsTdmArgumentSupported with a valid k_batch and K and the given addresses,
// with what it logged. It does not query the device, so acceptance is checked on any host.
template <typename Gemm>
std::pair<bool, std::string> tdm_pointers_ok(std::uintptr_t a, std::uintptr_t b, std::uintptr_t e)
{
    return logged([&] {
        return Gemm::Kernel::IsTdmArgumentSupported("MX GEMM",
                                                    1,
                                                    128,
                                                    reinterpret_cast<const void*>(a),
                                                    reinterpret_cast<const void*>(b),
                                                    reinterpret_cast<const void*>(e));
    });
}

// K values of the TDM gtests (1..4 x K_Tile and 512^3), the grouped MX TDM gtests (512 + 128 i) and
// the dispatcher bridge TDM tests with tile_k 128 (1, 2, 3, 4, 5, 8 x tile_k).
const std::vector<index_t> kTestedKsTileK128{128, 256, 384, 512, 640, 768, 896, 1024};

// M and N values the TDM gtests and the dispatcher bridge TDM tests use, including partial tiles.
const std::vector<index_t> kTestedMN{1, 17, 63, 64, 127, 257};

TEST(MxGemmTdmGates, AcceptsTestedShapes)
{
    for(index_t K : kTestedKsTileK128)
    {
        EXPECT_TRUE((shape_ok<F8, F8>(64, 64, K, K, K))) << K;
        EXPECT_TRUE((shape_ok<BF8, F8>(64, 64, K, K, K))) << K;
        EXPECT_TRUE((shape_ok<F4, F4>(64, 64, K, K, K))) << K;
        EXPECT_TRUE((shape_ok<F8, F4>(64, 64, K, K, K))) << K;
        EXPECT_TRUE((shape_ok<F8, F8, Row, Col, 128, 16>(64, 64, K, K, K))) << K;
        // The RRR and CRR rows of the TDM gtests, with dense strides.
        EXPECT_TRUE((shape_ok<F4, F4, Row, Row>(64, 64, K, K, 64))) << K;
        EXPECT_TRUE((shape_ok<F4, F8, Col, Row>(64, 64, K, 64, 64))) << K;
    }
    // The dispatcher bridge TDM tests with tile_k 256.
    for(index_t K : {256, 512, 768, 1024, 1280, 2048})
    {
        EXPECT_TRUE((shape_ok<F8, F8, Row, Col, 256>(64, 64, K, K, K))) << K;
    }
    // M and N are not contiguous in RCR, so their extents are not gated, packed or not.
    for(index_t MN : kTestedMN)
    {
        EXPECT_TRUE((shape_ok<F8, F8>(MN, MN, 256, 256, 256))) << MN;
        EXPECT_TRUE((shape_ok<F4, F4>(MN, MN, 256, 256, 256))) << MN;
    }
}

TEST(MxGemmTdmGates, RejectsNonPositiveK)
{
    EXPECT_FALSE((shape_ok<F8, F8>(64, 64, 0, 128, 128)));
    EXPECT_FALSE((shape_ok<F8, F8>(64, 64, -128, 128, 128)));
}

TEST(MxGemmTdmGates, RejectsKNotAMultipleOfThePackedScale)
{
    // One packed int32 of scale32 values covers 4 x 32 = 128 K; KPerBlock 32 isolates this check.
    EXPECT_FALSE((shape_ok<F8, F8, Row, Col, 32>(64, 64, 96, 96, 96)));
    EXPECT_FALSE((shape_ok<F8, F8, Row, Col, 32>(64, 64, 160, 160, 160)));
    EXPECT_FALSE((shape_ok<F8, F8, Row, Col, 16>(64, 64, 80, 80, 80))); // not even K % 32
    // scale16 packs 4 x 16 = 64 K.
    EXPECT_FALSE((shape_ok<F8, F8, Row, Col, 32, 16>(64, 64, 96, 96, 96)));
    EXPECT_TRUE((shape_ok<F8, F8, Row, Col, 64, 16>(64, 64, 192, 192, 192)));
    // The same gates apply to 6-bit operands.
    EXPECT_FALSE((shape_ok<F6, F6, Row, Col, 32>(64, 64, 96, 96, 96)));
    EXPECT_FALSE((shape_ok<BF6, BF6, Row, Col, 32, 16>(64, 64, 96, 96, 96)));
    EXPECT_TRUE((shape_ok<BF6, BF6, Row, Col, 64, 16>(64, 64, 192, 192, 192)));
}

TEST(MxGemmTdmGates, RejectsKNotAMultipleOfKPerBlock)
{
    EXPECT_FALSE((shape_ok<F8, F8, Row, Col, 256>(64, 64, 384, 384, 384)));
    EXPECT_FALSE((shape_ok<F8, F8, Row, Col, 256>(64, 64, 640, 640, 640)));
    EXPECT_TRUE((shape_ok<F8, F8, Row, Col, 256>(64, 64, 512, 512, 512)));
    EXPECT_FALSE((shape_ok<F6, F6, Row, Col, 256>(64, 64, 384, 384, 384)));
}

TEST(MxGemmTdmGates, RejectsPackedStridesOffAPackBoundary)
{
    constexpr index_t K = 512;
    // 6-bit packs hold 16 elements. 544 is a multiple of 16 but not of 12; 516 and 540 are
    // multiples of 12 but not of 16, so a check on the 12-byte size instead of PackedSize fails.
    for(index_t ok : {528, 544})
    {
        EXPECT_TRUE((shape_ok<F6, F6>(64, 64, K, ok, ok))) << ok;
        EXPECT_TRUE((shape_ok<BF6, BF6>(64, 64, K, ok, ok))) << ok;
    }
    for(index_t bad : {516, 520, 540})
    {
        EXPECT_FALSE((shape_ok<F6, F6>(64, 64, K, bad, 544))) << bad;
        EXPECT_FALSE((shape_ok<F6, F6>(64, 64, K, 544, bad))) << bad;
        EXPECT_FALSE((shape_ok<BF6, BF6>(64, 64, K, bad, 544))) << bad;
        EXPECT_FALSE((shape_ok<BF6, BF6>(64, 64, K, 544, bad))) << bad;
    }
    // The 32-element 6-bit packs need multiples of 32.
    EXPECT_FALSE((shape_ok<F6x32, F6x32>(64, 64, K, 528, 544)));
    EXPECT_TRUE((shape_ok<F6x32, F6x32>(64, 64, K, 544, 544)));
    // FP4 packs hold 2 elements.
    EXPECT_FALSE((shape_ok<F4, F4>(64, 64, K, 513, 514)));
    EXPECT_FALSE((shape_ok<F4, F4>(64, 64, K, 514, 513)));
    EXPECT_TRUE((shape_ok<F4, F4>(64, 64, K, 514, 514)));
    // Mixed pairs check each packed side; unpacked strides are unconstrained.
    EXPECT_TRUE((shape_ok<F6, F8>(64, 64, K, 528, 513)));
    EXPECT_FALSE((shape_ok<F6, F8>(64, 64, K, 520, 528)));
    EXPECT_TRUE((shape_ok<F8, F6>(64, 64, K, 513, 528)));
    EXPECT_FALSE((shape_ok<F8, F6>(64, 64, K, 528, 520)));
    EXPECT_TRUE((shape_ok<BF6, F8>(64, 64, K, 528, 513)));
    EXPECT_FALSE((shape_ok<F6, F4>(64, 64, K, 528, 513)));
    EXPECT_FALSE((shape_ok<F6, BF6>(64, 64, K, 528, 520)));
    EXPECT_TRUE((shape_ok<F8, F8>(64, 64, K, 513, 513)));
}

TEST(MxGemmTdmGates, RejectsPackedContiguousExtentsOffAPackBoundary)
{
    constexpr index_t K = 256;
    // A column-major packed A has M as its contiguous dimension; even strides isolate the extent.
    for(index_t M : {1, 3, 63, 257})
    {
        EXPECT_FALSE((shape_ok<F4, F8, Col, Row>(M, 64, K, 258, 64))) << M;
    }
    for(index_t M : {2, 64, 256})
    {
        EXPECT_TRUE((shape_ok<F4, F8, Col, Row>(M, 64, K, 258, 64))) << M;
    }
    EXPECT_FALSE((shape_ok<F6, F6, Col, Col>(24, 64, K, 528, 528)));
    EXPECT_TRUE((shape_ok<F6, F6, Col, Col>(32, 64, K, 528, 528)));
    // A row-major packed B has N as its contiguous dimension.
    EXPECT_FALSE((shape_ok<F8, F4, Row, Row>(64, 63, K, K, 258)));
    EXPECT_TRUE((shape_ok<F8, F4, Row, Row>(64, 62, K, K, 258)));
    // M of a row-major A and N of a column-major B are not contiguous, so they are not gated.
    EXPECT_TRUE((shape_ok<F4, F4, Row, Col>(63, 63, K, K, K)));
}

// The kernel-level checks: IsTdmShapeSupported forwards the kernel's types, layouts and tile
// constants, and IsSupportedArgument rejects for the shape reason, which it logs. Without a gfx125
// device IsSupportedArgument would also fail its device check, so the logged reason is what shows
// that the shape gate ran; the device check, which runs last, must not be reached.
template <typename Gemm>
void expect_shape_rejection(
    index_t M, index_t N, index_t K, index_t stride_A, index_t stride_B, const std::string& reason)
{
    using Kernel     = typename Gemm::Kernel;
    const auto kargs = Gemm::make_args(M, N, K, stride_A, stride_B);
    EXPECT_FALSE(Kernel::IsTdmShapeSupported(kargs)) << M << " " << N << " " << K;
    const auto [supported, log] = logged([&] { return Kernel::IsSupportedArgument(kargs); });
    EXPECT_FALSE(supported) << M << " " << N << " " << K;
    EXPECT_NE(log.find(reason), std::string::npos) << log;
    EXPECT_EQ(log.find("requires a gfx125 device"), std::string::npos) << log;
}

// Accepting an argument in IsSupportedArgument also needs a gfx125 device; without one only the
// shape part is checked here, and the calling test reports SKIPPED.
template <typename Gemm>
void expect_shape_accepted(index_t M, index_t N, index_t K, index_t stride_A, index_t stride_B)
{
    using Kernel     = typename Gemm::Kernel;
    const auto kargs = Gemm::make_args(M, N, K, stride_A, stride_B);
    EXPECT_TRUE(Kernel::IsTdmShapeSupported(kargs)) << M << " " << N << " " << K;
    if(ck_tile::is_gfx125_supported())
    {
        EXPECT_TRUE(Kernel::IsSupportedArgument(kargs)) << M << " " << N << " " << K;
    }
}

void skip_without_gfx125()
{
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Accepting an argument also requires a gfx125 device";
    }
}

template <typename Gemm>
void check_rcr_kernel_gates()
{
    for(index_t MN : kTestedMN)
    {
        for(index_t K : kTestedKsTileK128)
        {
            expect_shape_accepted<Gemm>(MN, MN, K, K, K);
        }
    }
    expect_shape_rejection<Gemm>(64, 64, 0, 128, 128, "requires K > 0");
    for(index_t K : {32, 64, 96, 160, 200})
    {
        expect_shape_rejection<Gemm>(
            64, 64, K, K, K, "requires K to be a multiple of ScaleBlockSize * KXdlPack = 128");
    }
}

// The kernel forwards FP6's 16-element pack to the stride gate: 520 is a multiple of 8 but not of
// 16, so an FP8 kernel would accept it.
template <typename Gemm>
void check_f6_kernel_stride_gates()
{
    expect_shape_rejection<Gemm>(
        64,
        64,
        512,
        520,
        512,
        "requires the stride of a packed A to be a multiple of its pack size 16");
    expect_shape_rejection<Gemm>(
        64,
        64,
        512,
        512,
        520,
        "requires the stride of a packed B to be a multiple of its pack size 16");
}

TEST(MxGemmTdmGates, KernelF8TdmV1)
{
    check_rcr_kernel_gates<TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV1, F8, F8>>();
    skip_without_gfx125();
}

TEST(MxGemmTdmGates, KernelF8TdmV2)
{
    check_rcr_kernel_gates<TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV2, F8, F8>>();
    skip_without_gfx125();
}

TEST(MxGemmTdmGates, KernelF4TdmV1)
{
    check_rcr_kernel_gates<TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV1, F4, F4>>();
    skip_without_gfx125();
}

TEST(MxGemmTdmGates, KernelF4TdmV2)
{
    check_rcr_kernel_gates<TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV2, F4, F4>>();
    skip_without_gfx125();
}

TEST(MxGemmTdmGates, KernelF6TdmV1)
{
    using Gemm = TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV1, F6, F6>;
    check_rcr_kernel_gates<Gemm>();
    check_f6_kernel_stride_gates<Gemm>();
    skip_without_gfx125();
}

TEST(MxGemmTdmGates, KernelF6TdmV2)
{
    using Gemm = TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV2, F6, F6>;
    check_rcr_kernel_gates<Gemm>();
    check_f6_kernel_stride_gates<Gemm>();
    skip_without_gfx125();
}

TEST(MxGemmTdmGates, KernelPassesItsKTile)
{
    // A 256-deep K tile: K = 384 and 640 pass the packed-scale check but not K % KPerBlock.
    using Gemm = TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV1, F8, F8, Row, Col, E8M0, E8M0, 256>;
    for(index_t K : {512, 768, 1024})
    {
        expect_shape_accepted<Gemm>(64, 64, K, K, K);
    }
    for(index_t K : {384, 640})
    {
        expect_shape_rejection<Gemm>(
            64, 64, K, K, K, "requires K to be a multiple of KPerBlock = 256");
    }
    skip_without_gfx125();
}

TEST(MxGemmTdmGates, KernelPassesItsLayouts)
{
    // The column-major FP4 A row of the TDM gtests (CRR, 32x32 warp tile): M is A's contiguous
    // dimension, so an odd M is rejected; with the dense stride M, so is the stride.
    using Gemm =
        TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV1, F4, F8, Col, Row, E5M3, E8M0, 128, 32>;
    for(index_t M : {2, 64, 256})
    {
        expect_shape_accepted<Gemm>(M, 64, 256, M, 64);
    }
    expect_shape_rejection<Gemm>(
        1, 64, 256, 1, 64, "requires the stride of a packed A to be a multiple of its pack size 2");
    expect_shape_rejection<Gemm>(
        63, 64, 256, 64, 64, "requires M of a column-major packed A to be a multiple");
    skip_without_gfx125();
}

TEST(MxGemmTdmGates, KernelPointerAlignment)
{
    // 1-byte A and B (FP8, and FP4 packs) take any address; the fp16 E needs an even one.
    using F8Gemm = TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV1, F8, F8>;
    using F4Gemm = TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV1, F4, F4>;
    for(std::uintptr_t offset : {0, 1, 2, 3})
    {
        const std::uintptr_t ab = kDeviceAddress + offset;
        EXPECT_TRUE(tdm_pointers_ok<F8Gemm>(ab, ab, kDeviceAddress).first) << offset;
        EXPECT_TRUE(tdm_pointers_ok<F4Gemm>(ab, ab, kDeviceAddress).first) << offset;
    }
    EXPECT_TRUE(tdm_pointers_ok<F8Gemm>(kDeviceAddress, kDeviceAddress, kDeviceAddress + 2).first);
    for(std::uintptr_t offset : {1, 3})
    {
        const auto [ok, log] =
            tdm_pointers_ok<F8Gemm>(kDeviceAddress, kDeviceAddress, kDeviceAddress + offset);
        EXPECT_FALSE(ok) << offset;
        EXPECT_NE(log.find(kAlignmentReason), std::string::npos) << log;
    }
}

TEST(MxGemmTdmGates, KernelF6PointerAlignment)
{
    // 12-byte FP6 packs need 4-byte aligned A and B, not 12-byte aligned ones: kDeviceAddress is 4
    // mod 12, and kDeviceAddress + 4 is 8 mod 12.
    using F6Gemm = TdmMxGemm<ck_tile::GemmPipelineAgBgCrCompTDMV1, F6, F6>;
    ASSERT_EQ(kDeviceAddress % sizeof(F6), 4u)
        << "the accepted addresses must not be 12-byte aligned";
    for(std::uintptr_t offset : {0, 4})
    {
        const std::uintptr_t ab = kDeviceAddress + offset;
        EXPECT_TRUE(tdm_pointers_ok<F6Gemm>(ab, ab, kDeviceAddress).first) << offset;
    }
    for(std::uintptr_t offset : {1, 2, 3})
    {
        const std::uintptr_t bad = kDeviceAddress + offset;
        const auto [a_ok, a_log] = tdm_pointers_ok<F6Gemm>(bad, kDeviceAddress, kDeviceAddress);
        EXPECT_FALSE(a_ok) << offset;
        EXPECT_NE(a_log.find(kAlignmentReason), std::string::npos) << a_log;
        const auto [b_ok, b_log] = tdm_pointers_ok<F6Gemm>(kDeviceAddress, bad, kDeviceAddress);
        EXPECT_FALSE(b_ok) << offset;
        EXPECT_NE(b_log.find(kAlignmentReason), std::string::npos) << b_log;
    }
}

template <typename Policy>
void expect_dense_f6_lds()
{
    using Problem = F6Problem;
    using Base    = typename Policy::Base;
    // The base policy pads these A and B rows, so the TDM policy is what removes the padding.
    EXPECT_EQ(lds_padding(Base::template GetLdsPaddingConfig<Problem, true>())[0], 1);
    EXPECT_EQ(lds_padding(Base::template GetLdsPaddingConfig<Problem, false>())[0], 1);
    const std::array<index_t, 3> no_padding{0, 0, 0};
    EXPECT_EQ(lds_padding(Policy::template GetLdsPaddingConfig<Problem, true>()), no_padding);
    EXPECT_EQ(lds_padding(Policy::template GetLdsPaddingConfig<Problem, false>()), no_padding);
    EXPECT_EQ(Policy::template GetSmemSize<Problem>(), kF6DenseLdsBufferBytes);
}

// 6-bit LDS rows are never padded, in the host pass as in the gfx125 device pass, so the host's LDS
// size is that of the dense rows.
TEST(MxGemmTdmGates, PolicyF6LdsRowsAreDense)
{
    expect_dense_f6_lds<F6PolicyV1>();
    expect_dense_f6_lds<F6PolicyV2>();
}

// An FP6 pipeline allocates two dense LDS buffers, because the TDM pipelines double-buffer LDS.
// Its kernel allocates the larger of the pipeline's and the TDM epilogue's LDS, which is the
// pipeline's.
template <template <typename...> class Pipeline>
void expect_dense_f6_kernel_lds()
{
    using Gemm = TdmMxGemm<Pipeline, F6, F6>;
    EXPECT_EQ(Pipeline<typename Gemm::Problem>::GetSmemSize(), 2 * kF6DenseLdsBufferBytes);
    EXPECT_EQ(Gemm::Kernel::GetSmemSize(), 2 * kF6DenseLdsBufferBytes);
}

TEST(MxGemmTdmGates, KernelF6LdsRowsAreDense)
{
    expect_dense_f6_kernel_lds<ck_tile::GemmPipelineAgBgCrCompTDMV1>();
    expect_dense_f6_kernel_lds<ck_tile::GemmPipelineAgBgCrCompTDMV2>();
}

// Every other operand type keeps the base policy's LDS padding in the host pass; only the gfx125
// device pass may drop it (IsTDMLdsPaddingConsistent).
template <typename Policy, typename A, typename B>
void expect_base_lds_padding()
{
    using Problem     = typename TdmMxProblem<A, B>::type;
    using Base        = typename Policy::Base;
    const auto base_a = lds_padding(Base::template GetLdsPaddingConfig<Problem, true>());
    const auto base_b = lds_padding(Base::template GetLdsPaddingConfig<Problem, false>());
    EXPECT_EQ(base_a[0], 1);
    EXPECT_EQ(base_b[0], 1);
    EXPECT_EQ(lds_padding(Policy::template GetLdsPaddingConfig<Problem, true>()), base_a);
    EXPECT_EQ(lds_padding(Policy::template GetLdsPaddingConfig<Problem, false>()), base_b);
}

TEST(MxGemmTdmGates, PolicyKeepsBaseLdsPaddingOnTheHost)
{
    expect_base_lds_padding<TdmPolicy<ck_tile::GemmPipelineAgBgCrCompTDMV1, F8, F8>, F8, F8>();
    expect_base_lds_padding<TdmPolicy<ck_tile::GemmPipelineAgBgCrCompTDMV1, F4, F4>, F4, F4>();
}

} // namespace
