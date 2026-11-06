// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched.hpp"

#include "gtest/gtest.h"

#include <numeric>
#include <vector>

namespace {

template <ck_tile::index_t M,
          bool Mask,
          bool StoreLSE                            = false,
          typename DataType                        = ck_tile::bf16_t,
          ck_tile::BlockAttentionBiasEnum BiasEnum = ck_tile::BlockAttentionBiasEnum::NO_BIAS,
          bool HasSink                             = false,
          ck_tile::index_t N                       = 64>
using MappingProblem = ck_tile::BlockFmhaPipelineProblem<
    DataType,
    DataType,
    DataType,
    float,
    float,
    DataType,
    uint8_t,
    float,
    DataType,
    float,
    DataType,
    ck_tile::TileFmhaShape<ck_tile::sequence<M, N, 32, 128, 32, 128>,
                           ck_tile::sequence<4, 1, 1>,
                           ck_tile::sequence<16, 16, 32>,
                           ck_tile::sequence<4, 1, 1>,
                           ck_tile::sequence<16, 16, 32>,
                           true>,
    false,
    ck_tile::ComposedAttention<0>,
    ck_tile::SimplifiedGenericAttentionMask<Mask>,
    false,
    ck_tile::TileFmhaTraits<false,
                            false,
                            false,
                            false,
                            false,
                            BiasEnum,
                            false,
                            StoreLSE,
                            false,
                            ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                            -1,
                            false,
                            HasSink>,
    true,
    true>;

template <ck_tile::index_t M,
          bool Mask,
          bool StoreLSE                            = false,
          typename DataType                        = ck_tile::bf16_t,
          ck_tile::BlockAttentionBiasEnum BiasEnum = ck_tile::BlockAttentionBiasEnum::NO_BIAS,
          bool HasSink                             = false>
using MappingKernel = ck_tile::FmhaFwdKernel<
    ck_tile::BlockFmhaPipelineQRKSVSTdm<
        MappingProblem<M, Mask, StoreLSE, DataType, BiasEnum, HasSink>>,
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, DataType, false, false>>>;

// Which pipelines and tiles get the K/V NT_HT hint: only the fields the predicate
// reads, so every axis can be pinned without instantiating a full pipeline.
template <typename KDataType_, ck_tile::index_t M0, ck_tile::index_t HDim, bool LdsArena = true>
struct HintProbePipeline
{
    static constexpr bool kUsesLdsArena   = LdsArena;
    static constexpr ck_tile::index_t kM0 = M0;
    using KDataType                       = KDataType_;
    struct BlockFmhaShape
    {
        static constexpr ck_tile::index_t kQKHeaddim = HDim;
    };
};

template <typename Pipeline>
constexpr bool kKVHint = ck_tile::detail::uses_kv_nt_ht_hint_v<Pipeline>;

static_assert(kKVHint<HintProbePipeline<ck_tile::bf16_t, 64, 128>>);
static_assert(kKVHint<HintProbePipeline<ck_tile::half_t, 64, 128>>);
static_assert(kKVHint<HintProbePipeline<ck_tile::bf16_t, 16, 128>>);
static_assert(kKVHint<HintProbePipeline<ck_tile::bf16_t, 64, 64>>);
static_assert(kKVHint<HintProbePipeline<ck_tile::bf16_t, 64, 192>>);
static_assert(!kKVHint<HintProbePipeline<ck_tile::bf16_t, 128, 128>>);
static_assert(!kKVHint<HintProbePipeline<ck_tile::bf16_t, 64, 256>>);
static_assert(!kKVHint<HintProbePipeline<ck_tile::fp8_t, 64, 128>>);
static_assert(!kKVHint<HintProbePipeline<ck_tile::bf16_t, 64, 128, false>>);
static_assert(kKVHint<ck_tile::BlockFmhaPipelineQRKSVSTdm<MappingProblem<64, true>>>);
static_assert(!kKVHint<ck_tile::BlockFmhaPipelineQRKSVSTdm<MappingProblem<128, true>>>);

using ScheduledMappingKernel = ck_tile::FmhaFwdKernel<
    ck_tile::BlockFmhaPipelineQRKSVSTdmSched<
        MappingProblem<128,
                       true,
                       false,
                       ck_tile::bf16_t,
                       ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                       false,
                       128>>,
    ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<float, ck_tile::bf16_t, false, false>>>;

// Bias, dropout, softcap and quantization are rejected by the scheduled pipeline
// itself. These probes inherit a valid pipeline and override only eligibility
// facts; they exercise the real FmhaFwdKernel predicate without GPU computation.
template <bool Lse                             = false,
          bool Dropout                         = false,
          bool Sink                            = false,
          bool Softcap                         = false,
          ck_tile::BlockAttentionBiasEnum Bias = ck_tile::BlockAttentionBiasEnum::NO_BIAS,
          ck_tile::BlockAttentionQuantScaleEnum Scale =
              ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE>
using EligibilityTraits = ck_tile::TileFmhaTraits<false,
                                                  false,
                                                  false,
                                                  false,
                                                  Softcap,
                                                  Bias,
                                                  false,
                                                  Lse,
                                                  Dropout,
                                                  Scale,
                                                  -1,
                                                  false,
                                                  Sink>;

template <typename Traits_         = EligibilityTraits<>,
          typename Q               = ck_tile::bf16_t,
          typename K               = Q,
          typename V               = Q,
          ck_tile::index_t M       = 128,
          ck_tile::index_t N       = 128,
          ck_tile::index_t HeadDim = 128,
          bool Mask                = true,
          bool Group               = false>
struct ScheduledEligibilityPipeline : ScheduledMappingKernel::FmhaPipeline
{
    struct Problem : ScheduledMappingKernel::FmhaPipeline::Problem
    {
        static constexpr auto QScaleEnum = Traits_::QScaleEnum;
    };
    using QDataType = Q;
    using KDataType = K;
    using VDataType = V;
    using FmhaMask  = ck_tile::SimplifiedGenericAttentionMask<Mask>;
    struct BlockFmhaShape : ScheduledMappingKernel::FmhaPipeline::BlockFmhaShape
    {
        static constexpr auto kQKHeaddim = HeadDim;
    };
    static constexpr auto kM0               = M;
    static constexpr auto kN0               = N;
    static constexpr bool kIsGroupMode      = Group;
    static constexpr bool kStoreLSE         = Traits_::kStoreLSE;
    static constexpr bool kHasDropout       = Traits_::kHasDropout;
    static constexpr bool kHasSink          = Traits_::kHasSink;
    static constexpr bool kHasLogitsSoftCap = Traits_::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = Traits_::BiasEnum;
};

template <typename Pipeline>
constexpr bool kSupportsXcdCausal =
    ck_tile::FmhaFwdKernel<Pipeline,
                           ScheduledMappingKernel::EpiloguePipeline>::kSupportsXcdCausalOrder;

using ck_tile::bf16_t;
using ck_tile::BlockAttentionBiasEnum;
using ck_tile::BlockAttentionQuantScaleEnum;
using ck_tile::fp16_t;
using DefaultEligibilityTraits = EligibilityTraits<>;

// Pin each exclusion independently: removing any one gate must fail compilation.
static_assert(ScheduledMappingKernel::kSupportsXcdCausalOrder);
static_assert(kSupportsXcdCausal<ScheduledEligibilityPipeline<>>);
static_assert(!kSupportsXcdCausal<ScheduledEligibilityPipeline<EligibilityTraits<true>>>);
static_assert(!kSupportsXcdCausal<ScheduledEligibilityPipeline<EligibilityTraits<false, true>>>);
static_assert(
    !kSupportsXcdCausal<ScheduledEligibilityPipeline<EligibilityTraits<false, false, true>>>);
static_assert(!kSupportsXcdCausal<
              ScheduledEligibilityPipeline<EligibilityTraits<false, false, false, true>>>);
static_assert(
    !kSupportsXcdCausal<ScheduledEligibilityPipeline<
        EligibilityTraits<false, false, false, false, BlockAttentionBiasEnum::ELEMENTWISE_BIAS>>>);
static_assert(
    !kSupportsXcdCausal<
        ScheduledEligibilityPipeline<EligibilityTraits<false,
                                                       false,
                                                       false,
                                                       false,
                                                       BlockAttentionBiasEnum::NO_BIAS,
                                                       BlockAttentionQuantScaleEnum::PERTENSOR>>>);
static_assert(!kSupportsXcdCausal<
              ScheduledEligibilityPipeline<DefaultEligibilityTraits, fp16_t, bf16_t, bf16_t>>);
static_assert(!kSupportsXcdCausal<
              ScheduledEligibilityPipeline<DefaultEligibilityTraits, bf16_t, fp16_t, bf16_t>>);
static_assert(!kSupportsXcdCausal<
              ScheduledEligibilityPipeline<DefaultEligibilityTraits, bf16_t, bf16_t, fp16_t>>);
template <ck_tile::index_t M       = 128,
          ck_tile::index_t N       = 128,
          ck_tile::index_t HeadDim = 128,
          bool Mask                = true,
          bool Group               = false>
using GeometryEligibilityPipeline = ScheduledEligibilityPipeline<DefaultEligibilityTraits,
                                                                 bf16_t,
                                                                 bf16_t,
                                                                 bf16_t,
                                                                 M,
                                                                 N,
                                                                 HeadDim,
                                                                 Mask,
                                                                 Group>;

static_assert(!kSupportsXcdCausal<GeometryEligibilityPipeline<64>>);
static_assert(!kSupportsXcdCausal<GeometryEligibilityPipeline<128, 64>>);
static_assert(!kSupportsXcdCausal<GeometryEligibilityPipeline<128, 128, 64>>);
static_assert(!kSupportsXcdCausal<GeometryEligibilityPipeline<128, 128, 192>>);
static_assert(!kSupportsXcdCausal<GeometryEligibilityPipeline<128, 128, 128, false>>);
static_assert(!kSupportsXcdCausal<GeometryEligibilityPipeline<128, 128, 128, true, true>>);
// Ordinary qr_tdm remains excluded even with M128/N128/D128.
static_assert(
    !kSupportsXcdCausal<ck_tile::BlockFmhaPipelineQRKSVSTdm<
        MappingProblem<128, true, false, bf16_t, BlockAttentionBiasEnum::NO_BIAS, false, 128>>>);

template <typename Kernel>
__global__ void capture_mapping(typename Kernel::Kargs args, int* output)
{
    if(threadIdx.x == 0)
    {
        const auto [q, v, head, batch] = Kernel::GetTileIndex(args);
        const auto linear      = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        output[linear * 4 + 0] = q;
        output[linear * 4 + 1] = v;
        output[linear * 4 + 2] = head;
        output[linear * 4 + 3] = batch;
    }
}

template <ck_tile::index_t M,
          bool Mask,
          bool StoreLSE                            = false,
          typename DataType                        = ck_tile::bf16_t,
          ck_tile::BlockAttentionBiasEnum BiasEnum = ck_tile::BlockAttentionBiasEnum::NO_BIAS,
          bool HasSink                             = false>
std::vector<int> capture(int heads,
                         int batches,
                         int tiles,
                         int left             = -1,
                         int extra_k          = 0,
                         int hdim_q           = 128,
                         int hdim_v           = 128,
                         bool bhsd            = true,
                         int variable_lengths = 0)
{
    using Kernel = MappingKernel<M, Mask, StoreLSE, DataType, BiasEnum, HasSink>;
    typename Kernel::Kargs args{};
    args.hdim_q         = hdim_q;
    args.hdim_v         = hdim_v;
    args.seqlen_q       = tiles * M;
    args.seqlen_k       = tiles * M + extra_k;
    args.stride_q       = bhsd ? hdim_q : heads * hdim_q;
    args.nhead_stride_q = bhsd ? args.seqlen_q * hdim_q : hdim_q;
    if constexpr(Mask)
    {
        args.window_size_left  = left;
        args.window_size_right = 0;
        args.mask_type         = ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT;
    }
    const auto grid = Kernel::GridSize(batches, heads, args.seqlen_q, args.hdim_v);
    std::vector<int> result(grid.x * grid.y * grid.z * 4);
    ck_tile::DeviceMem device(result.size() * sizeof(int));
    // GetTileIndex inspects these pointers but does not dereference them.
    if(variable_lengths == 1)
        args.cu_seqlen_q_ptr = static_cast<int*>(device.GetDeviceBuffer());
    if(variable_lengths == 2)
        args.cu_seqlen_k_ptr = static_cast<int*>(device.GetDeviceBuffer());
    hipLaunchKernelGGL(capture_mapping<Kernel>,
                       grid,
                       dim3(32),
                       0,
                       0,
                       args,
                       static_cast<int*>(device.GetDeviceBuffer()));
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    device.FromDevice(result.data());
    return result;
}

TEST(QrTdmCausalMapping, CausalLongTilesComeFirstAcrossHeads)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    // A head-major implementation restarts at the longest tile for each head;
    // this observable schedule must instead keep long causal work together.
    const auto actual = capture<128, true>(2, 1, 4);
    const std::vector<int> expected{3, 0, 0, 0, 3, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1, 0,
                                    1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    EXPECT_EQ(actual, expected);
}

TEST(QrTdmCausalMapping, DenseRetainsHeadMajorOrder)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const auto actual = capture<128, false>(2, 1, 3);
    const std::vector<int> expected{0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0,
                                    0, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0};
    EXPECT_EQ(actual, expected);
}

TEST(QrTdmCausalMapping, OtherMasksAndRectangularInputsRetainTheirSchedule)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const std::vector<int> expected{2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                    2, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0};
    EXPECT_EQ((capture<128, true>(2, 1, 3, 127)), expected);
    EXPECT_EQ((capture<128, true>(2, 1, 3, -1, 128)), expected);
}

TEST(QrTdmCausalMapping, M64CoversEachBatchHeadAndTileExactlyOnce)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const auto actual = capture<64, true>(8, 2, 6);
    std::vector<int> visits(2 * 8 * 6);
    for(std::size_t i = 0; i < actual.size(); i += 4)
    {
        const int q = actual[i], v = actual[i + 1], h = actual[i + 2], b = actual[i + 3];
        ASSERT_GE(q, 0);
        ASSERT_LT(q, 6);
        ASSERT_EQ(v, 0);
        ASSERT_GE(h, 0);
        ASSERT_LT(h, 8);
        ASSERT_GE(b, 0);
        ASSERT_LT(b, 2);
        ++visits[(b * 8 + h) * 6 + q];
    }
    for(int count : visits)
        EXPECT_EQ(count, 1);
}

TEST(QrTdmCausalMapping, SupportedVariantsUseQMajorOrder)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const std::vector<int> expected{2, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0,
                                    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    EXPECT_EQ((capture<128, true, true>(2, 1, 3)), expected);
    EXPECT_EQ((capture<128, true, false, ck_tile::half_t>(2, 1, 3)), expected);
    EXPECT_EQ((capture<128,
                       true,
                       false,
                       ck_tile::bf16_t,
                       ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS>(2, 1, 3)),
              expected);
    EXPECT_EQ(
        (capture<128, true, false, ck_tile::bf16_t, ck_tile::BlockAttentionBiasEnum::NO_BIAS, true>(
            2, 1, 3)),
        expected);
}

TEST(QrTdmCausalMapping, SingleOutputTileVariantsUseQMajorOrder)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const std::vector<int> expected{2, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0,
                                    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    EXPECT_EQ((capture<64, true>(2, 1, 3)), expected);
    EXPECT_EQ((capture<128, true>(2, 1, 3, -1, 0, 64, 128)), expected);
    EXPECT_EQ((capture<128, true>(2, 1, 3, -1, 0, 128, 64)), expected);
}

TEST(QrTdmCausalMapping, BshdMultipleOutputTilesReverseOnlyQ)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    // The masked XCD threshold selects the direct grid for this small BSHD case.
    const std::vector<int> expected{2, 0, 0, 0, 2, 0, 1, 0, 2, 1, 0, 0, 2, 1, 1, 0,
                                    1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                                    0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0};
    EXPECT_EQ((capture<128, true>(2, 1, 3, -1, 0, 128, 256, false)), expected);
}

TEST(QrTdmCausalMapping, BshdMultipleOutputTilesCoverTheXcdGrid)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    // At the default eight XCDs this meets the masked remap threshold exactly:
    // 48 heads == 8 XCDs * (3 Q tiles * 2 V tiles).
    const auto actual = capture<128, true>(48, 1, 3, -1, 0, 128, 256, false);
    ASSERT_EQ(actual.size(), 48 * 3 * 2 * 4);
    EXPECT_EQ(actual[0], 2);
    std::vector<int> visits(48 * 3 * 2);
    for(std::size_t i = 0; i < actual.size(); i += 4)
    {
        const int q = actual[i], v = actual[i + 1], head = actual[i + 2], batch = actual[i + 3];
        ASSERT_GE(q, 0);
        ASSERT_LT(q, 3);
        ASSERT_GE(v, 0);
        ASSERT_LT(v, 2);
        ASSERT_GE(head, 0);
        ASSERT_LT(head, 48);
        ASSERT_EQ(batch, 0);
        ASSERT_EQ(++visits[(head * 3 + q) * 2 + v], 1);
    }
}

// The default XCD count is a device-pass macro; the host pass sees zero.
// Read the compiled value so this test also respects an explicit build override.
__global__ void capture_xcd_count(int* count) { *count = CK_TILE_XCD_REMAP_NUM_XCDS; }

TEST(QrTdmCausalMapping, MultipleOutputTilesUseHeadMajorDecoderWithXcdRemap)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const std::vector<int> head_major{2, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 2, 1, 1, 0,
                                      1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0};
    ck_tile::DeviceMem count_device(sizeof(int));
    hipLaunchKernelGGL(capture_xcd_count,
                       dim3(1),
                       dim3(1),
                       0,
                       0,
                       static_cast<int*>(count_device.GetDeviceBuffer()));
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    int xcds = 0;
    count_device.FromDevice(&xcds);
    std::vector<int> expected(head_major.size());
    constexpr int blocks = 12;
    for(int l = 0; l < blocks; ++l)
    {
        // Independently enumerate each XCD's contiguous segment, then index the
        // fixed head-major fixture. Removing either the remap or Q reversal fails.
        int logical = l;
        if(xcds > 0)
        {
            const int xcd = l % xcds;
            logical       = l / xcds;
            for(int preceding = 0; preceding < xcd; ++preceding)
                logical += blocks / xcds + (preceding < blocks % xcds ? 1 : 0);
        }
        for(int i = 0; i < 4; ++i)
            expected[l * 4 + i] = head_major[logical * 4 + i];
    }
    EXPECT_EQ((capture<128, true>(2, 1, 3, -1, 0, 128, 256)), expected);
}

// Build the expected logical head-major stream, then distribute contiguous
// segments to XCDs. This does not call the production permutation or decoder.
std::vector<int> head_major_fixture(int heads, int tiles, int xcds)
{
    std::vector<int> logical;
    for(int head = 0; head < heads; ++head)
        for(int q = tiles - 1; q >= 0; --q)
            logical.insert(logical.end(), {q, 0, head, 0});
    if(xcds == 0)
        return logical;
    std::vector<int> physical(logical.size());
    const int blocks = heads * tiles;
    int source       = 0;
    for(int xcd = 0; xcd < xcds; ++xcd)
        for(int slot = xcd; slot < blocks; slot += xcds, ++source)
            for(int coord = 0; coord < 4; ++coord)
                physical[slot * 4 + coord] = logical[source * 4 + coord];
    return physical;
}

int compiled_xcd_count()
{
    ck_tile::DeviceMem device(sizeof(int));
    hipLaunchKernelGGL(
        capture_xcd_count, dim3(1), dim3(1), 0, 0, static_cast<int*>(device.GetDeviceBuffer()));
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    int count = 0;
    device.FromDevice(&count);
    return count;
}

TEST(QrTdmCausalMapping, MaskedXcdThresholdKeepsItsBoundary)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const int xcds = compiled_xcd_count();
    if(xcds == 0)
        GTEST_SKIP();
    constexpr int tiles = 4;
    for(int heads : {xcds * tiles - 1, xcds * tiles, xcds * tiles + 1})
    {
        SCOPED_TRACE(heads);
        std::vector<int> expected;
        if(heads < xcds * tiles)
            for(int q = tiles - 1; q >= 0; --q)
                for(int head = 0; head < heads; ++head)
                    expected.insert(expected.end(), {q, 0, head, 0});
        else
            expected = head_major_fixture(heads, tiles, xcds);
        EXPECT_EQ((capture<128, true>(heads, 1, tiles, -1, 0, 128, 128, false)), expected);
    }
}

TEST(QrTdmCausalMapping, VariableLengthsRetainGridAndPlainHeadMajorFallbacks)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const int xcds      = compiled_xcd_count();
    const int heads     = xcds > 0 ? xcds * 4 : 8;
    constexpr int tiles = 4;
    std::vector<int> grid;
    for(int q = tiles - 1; q >= 0; --q)
        for(int head = 0; head < heads; ++head)
            grid.insert(grid.end(), {q, 0, head, 0});
    for(int pointer : {1, 2})
    {
        SCOPED_TRACE(pointer);
        EXPECT_EQ((capture<128, true>(heads, 1, tiles, -1, 0, 128, 128, false, pointer)), grid);
        // A non-square BHSD mask bypasses the causal override. With variable
        // lengths it must retain plain head-major order, without XCD remapping.
        EXPECT_EQ((capture<128, true>(heads, 1, tiles, -1, 128, 128, 128, true, pointer)),
                  head_major_fixture(heads, tiles, 0));
    }
}

TEST(QrTdmCausalMapping, CausalOverridePrecedesUniformLengthCheck)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    const std::vector<int> expected{2, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0,
                                    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    for(int pointer : {1, 2})
        EXPECT_EQ((capture<128, true>(2, 1, 3, -1, 0, 128, 128, true, pointer)), expected);
}

std::vector<int> capture_scheduled(
    bool bhsd, int tiles = 4, int ratio = 1, int variable_lengths = 0, int batches = 128)
{
    using Kernel = ScheduledMappingKernel;
    typename Kernel::Kargs args{};
    args.hdim_q = args.hdim_v = 128;
    args.seqlen_q = args.seqlen_k = tiles * 128;
    args.num_head_q = args.num_head_q_total = 2;
    args.nhead_ratio_qk                     = ratio;
    args.stride_q                           = bhsd ? 128 : 256;
    args.nhead_stride_q                     = bhsd ? args.seqlen_q * 128 : 128;
    args.window_size_left                   = -1;
    args.window_size_right                  = 0;
    args.mask_type                          = ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT;
    const auto grid                         = Kernel::GridSize(batches, 2, args.seqlen_q, 128);
    std::vector<int> actual(batches * 2 * tiles * 4);
    ck_tile::DeviceMem device(actual.size() * sizeof(int));
    if(variable_lengths == 1)
        args.cu_seqlen_q_ptr = static_cast<int*>(device.GetDeviceBuffer());
    if(variable_lengths == 2)
        args.cu_seqlen_k_ptr = static_cast<int*>(device.GetDeviceBuffer());
    hipLaunchKernelGGL(capture_mapping<Kernel>,
                       grid,
                       dim3(32),
                       0,
                       0,
                       args,
                       static_cast<int*>(device.GetDeviceBuffer()));
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    device.FromDevice(actual.data());
    return actual;
}

// Runtime device names also accept gfx1250-strict, whose device pass does
// not enable the measured grouping. The paired opt-in also supersedes it.
// Query the compiled configuration instead of inferring it from a device name.
__global__ void capture_xcd_causal_enabled(int* supported)
{
#if defined(__gfx1250__) && \
    !(defined(CK_TILE_FMHA_PAIRED_XCC_ORDER) && CK_TILE_FMHA_PAIRED_XCC_ORDER)
    *supported = 1;
#else
    *supported = 0;
#endif
}

bool compiled_xcd_causal_enabled()
{
    ck_tile::DeviceMem device(sizeof(int));
    hipLaunchKernelGGL(capture_xcd_causal_enabled,
                       dim3(1),
                       dim3(1),
                       0,
                       0,
                       static_cast<int*>(device.GetDeviceBuffer()));
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    int supported = 0;
    device.FromDevice(&supported);
    return supported != 0;
}

TEST(QrTdmCausalMapping, ScheduledXcdCausalGroupsKeepWholeHeadsInBothLayouts)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    if(compiled_xcd_count() != 8 || !compiled_xcd_causal_enabled())
        GTEST_SKIP();
    for(bool bhsd : {true, false})
        for(int rows : {32, 64, 96, 256})
            for(int qtiles : {3, 4, 16})
            {
                SCOPED_TRACE(rows);
                SCOPED_TRACE(qtiles);
                const auto actual = capture_scheduled(bhsd, qtiles, 1, 0, rows / 2);
                std::vector<int> expected(actual.size());
                const int rows_per_xcd = rows / 8;
                const int group        = std::gcd(rows_per_xcd, 16);
                // Independent queue enumeration, without calling the mapper.
                for(int xcd = 0; xcd < 8; ++xcd)
                {
                    int slot = 0;
                    for(int first = xcd * rows_per_xcd; first < (xcd + 1) * rows_per_xcd;
                        first += group)
                        for(int q = qtiles - 1; q >= 0; --q)
                            for(int row = first; row < first + group; ++row, ++slot)
                            {
                                const int l         = xcd + slot * 8;
                                expected[4 * l]     = q;
                                expected[4 * l + 2] = row % 2;
                                expected[4 * l + 3] = row / 2;
                            }
                }
                EXPECT_EQ(actual, expected);
            }
}

TEST(QrTdmCausalMapping, ScheduledVarlenGqaAndLongSequencesRetainTheirFallbacks)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    if(compiled_xcd_count() != 8 || !compiled_xcd_causal_enabled())
        GTEST_SKIP();
    for(bool bhsd : {true, false})
        for(int variant = 0; variant < 4; ++variant)
        {
            const int tiles            = variant == 3 ? 17 : 4;
            const int ratio            = variant == 0 ? 2 : 1;
            const int variable_lengths = variant == 1 || variant == 2 ? variant : 0;
            const auto actual          = capture_scheduled(bhsd, tiles, ratio, variable_lengths);
            std::vector<int> expected(actual.size());
            if(bhsd || variable_lengths != 0)
            {
                int l = 0;
                for(int batch = 0; batch < 128; ++batch)
                    for(int q = tiles - 1; q >= 0; --q)
                        for(int head = 0; head < 2; ++head, ++l)
                        {
                            expected[4 * l]     = q;
                            expected[4 * l + 2] = head;
                            expected[4 * l + 3] = batch;
                        }
            }
            else
            {
                // Uniform BSHD retains contiguous head-major XCD segments.
                for(int xcd = 0; xcd < 8; ++xcd)
                {
                    int slot = 0;
                    for(int row = xcd * 32; row < (xcd + 1) * 32; ++row)
                        for(int q = tiles - 1; q >= 0; --q, ++slot)
                        {
                            const int l         = xcd + 8 * slot;
                            expected[4 * l]     = q;
                            expected[4 * l + 2] = row % 2;
                            expected[4 * l + 3] = row / 2;
                        }
                }
            }
            EXPECT_EQ(actual, expected);
        }
}

} // namespace
