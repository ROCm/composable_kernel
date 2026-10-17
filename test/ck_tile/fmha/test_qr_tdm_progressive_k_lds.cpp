// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/fmha/block/variants.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/device_memory.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <string>
#include <vector>

namespace qr_tdm_progressive_k_lds_test {

template <ck_tile::index_t M>
using TestFmhaShape = ck_tile::TileFmhaShape<ck_tile::sequence<M, 64, 32, 128, 32, 128>,
                                             ck_tile::sequence<4, 1, 1>,
                                             ck_tile::sequence<16, 16, 32>,
                                             ck_tile::sequence<4, 1, 1>,
                                             ck_tile::sequence<16, 16, 32>,
                                             true>;

using TestFmhaTraits = ck_tile::TileFmhaTraits<false,
                                               false,
                                               false,
                                               false,
                                               false,
                                               ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                               false,
                                               false,
                                               false,
                                               ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE>;

// StreamLLM sink tokens. Sink only changes the pipeline when the mask has a
// bounded left window, so the sink problems below pair it with a real mask.
using TestFmhaSinkTraits = ck_tile::TileFmhaTraits<false,
                                                   false,
                                                   false,
                                                   false,
                                                   false,
                                                   ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                   false,
                                                   false,
                                                   false,
                                                   ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                                   -1,
                                                   false,
                                                   /*kHasSink_=*/true>;

using TestCausalMask = ck_tile::SimplifiedGenericAttentionMask<true>;

template <ck_tile::index_t M,
          bool UseDoubleKVLdsBuffer = false,
          bool ProgressiveDsLoadK   = false,
          typename DataType         = ck_tile::half_t,
          typename Mask             = ck_tile::SimplifiedGenericAttentionMask<false>,
          typename Traits           = TestFmhaTraits>
using TestFmhaProblem = ck_tile::BlockFmhaPipelineProblem<DataType,
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
                                                          TestFmhaShape<M>,
                                                          false,
                                                          ck_tile::ComposedAttention<0>,
                                                          Mask,
                                                          false,
                                                          Traits,
                                                          UseDoubleKVLdsBuffer,
                                                          ProgressiveDsLoadK>;

using TestFmhaShapeD64 = ck_tile::TileFmhaShape<ck_tile::sequence<128, 64, 32, 32, 32, 64>,
                                                ck_tile::sequence<4, 1, 1>,
                                                ck_tile::sequence<16, 16, 32>,
                                                ck_tile::sequence<4, 1, 1>,
                                                ck_tile::sequence<16, 16, 32>,
                                                true,
                                                true>;

using TestFmhaShapeD256 = ck_tile::TileFmhaShape<ck_tile::sequence<64, 64, 32, 256, 32, 256>,
                                                 ck_tile::sequence<4, 1, 1>,
                                                 ck_tile::sequence<16, 16, 32>,
                                                 ck_tile::sequence<4, 1, 1>,
                                                 ck_tile::sequence<16, 16, 32>,
                                                 true,
                                                 true>;

using TestFmhaShapeD512 = ck_tile::TileFmhaShape<ck_tile::sequence<64, 64, 32, 512, 32, 512>,
                                                 ck_tile::sequence<4, 1, 1>,
                                                 ck_tile::sequence<16, 16, 32>,
                                                 ck_tile::sequence<4, 1, 1>,
                                                 ck_tile::sequence<16, 16, 32>,
                                                 true,
                                                 true>;

template <typename Shape,
          bool UseDoubleKVLdsBuffer = false,
          bool ProgressiveDsLoadK   = false,
          typename DataType         = ck_tile::half_t,
          typename Mask             = ck_tile::SimplifiedGenericAttentionMask<false>>
using TestFmhaProblemForShape = ck_tile::BlockFmhaPipelineProblem<DataType,
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
                                                                  Shape,
                                                                  false,
                                                                  ck_tile::ComposedAttention<0>,
                                                                  Mask,
                                                                  false,
                                                                  TestFmhaTraits,
                                                                  UseDoubleKVLdsBuffer,
                                                                  ProgressiveDsLoadK>;

using SingleBufferM64Problem = TestFmhaProblem<64>;
// Progressive K reload on the single-buffer path: the combination the codegen
// used to reject, and the one d256 and d512 run in production.
using SingleBufferProgressiveM64Problem = TestFmhaProblem<64, false, true>;
using DoubleBufferM64Problem            = TestFmhaProblem<64, true>;
using ProgressiveM64Problem             = TestFmhaProblem<64, true, true>;
using DoubleBufferM128Problem           = TestFmhaProblem<128, true>;
using ProgressiveM128Problem            = TestFmhaProblem<128, true, true>;
using ProgressiveM128D64Problem         = TestFmhaProblemForShape<TestFmhaShapeD64, true, true>;
using ProgressiveM64D256Problem         = TestFmhaProblemForShape<TestFmhaShapeD256, true, true>;
using SingleBufferSinkM64Problem =
    TestFmhaProblem<64, false, false, ck_tile::half_t, TestCausalMask, TestFmhaSinkTraits>;
using DoubleBufferSinkM64Problem =
    TestFmhaProblem<64, true, false, ck_tile::half_t, TestCausalMask, TestFmhaSinkTraits>;
// The shipped d256 and d512 configurations: single-V/double-K buffers, progressive
// reload, and the plain single-buffer counterpart to cross-check them against.
using SingleBufferD256Problem            = TestFmhaProblemForShape<TestFmhaShapeD256>;
using SingleBufferProgressiveD256Problem = TestFmhaProblemForShape<TestFmhaShapeD256, false, true>;
using SingleBufferD512Problem            = TestFmhaProblemForShape<TestFmhaShapeD512>;
using SingleBufferProgressiveD512Problem = TestFmhaProblemForShape<TestFmhaShapeD512, false, true>;

static_assert(!SingleBufferM64Problem::kUseDoubleKVLdsBuffer);
static_assert(!SingleBufferM64Problem::kProgressiveDsLoadK);
static_assert(!SingleBufferProgressiveM64Problem::kUseDoubleKVLdsBuffer);
static_assert(SingleBufferProgressiveM64Problem::kProgressiveDsLoadK);
static_assert(DoubleBufferM64Problem::kUseDoubleKVLdsBuffer);
static_assert(!DoubleBufferM64Problem::kProgressiveDsLoadK);
static_assert(ProgressiveM64Problem::kUseDoubleKVLdsBuffer);
static_assert(ProgressiveM64Problem::kProgressiveDsLoadK);
static_assert(DoubleBufferM128Problem::kUseDoubleKVLdsBuffer);
static_assert(!DoubleBufferM128Problem::kProgressiveDsLoadK);
static_assert(ProgressiveM128Problem::kUseDoubleKVLdsBuffer);
static_assert(ProgressiveM128Problem::kProgressiveDsLoadK);
static_assert(!SingleBufferSinkM64Problem::kUseDoubleKVLdsBuffer);
static_assert(DoubleBufferSinkM64Problem::kUseDoubleKVLdsBuffer);
static_assert(SingleBufferSinkM64Problem::FmhaMask::IsMasking);

template <typename Problem>
using TestPipeline = ck_tile::BlockFmhaPipelineQRKSVSTdm<Problem>;

static_assert(TestPipeline<SingleBufferSinkM64Problem>::kHasSink);
static_assert(TestPipeline<DoubleBufferSinkM64Problem>::kHasSink);

// Every path stages K as a whole tile now, which is what the progressive reload
// needs; the single-buffer one still never reaches the K-pair staging variant,
// whose shape requirements no qr_tdm single-buffer tile satisfies.
static_assert(TestPipeline<SingleBufferM64Problem>::kKLoadOnce);
static_assert(TestPipeline<SingleBufferProgressiveM64Problem>::kKLoadOnce);
static_assert(!TestPipeline<SingleBufferProgressiveM64Problem>::kStagedKPairs);
static_assert(!TestPipeline<SingleBufferProgressiveD256Problem>::kStagedKPairs);
static_assert(!TestPipeline<SingleBufferProgressiveD512Problem>::kStagedKPairs);
static_assert(TestPipeline<SingleBufferProgressiveD512Problem>::kKLoadOnce);
static_assert(TestPipeline<DoubleBufferM64Problem>::kKLoadOnce);
static_assert(TestPipeline<DoubleBufferM128Problem>::kKLoadOnce);
static_assert(TestPipeline<ProgressiveM64Problem>::kKLoadOnce);
static_assert(TestPipeline<ProgressiveM128Problem>::kKLoadOnce);
static_assert(!TestPipeline<DoubleBufferM128Problem>::kStagedKPairs);
static_assert(!TestPipeline<ProgressiveM64Problem>::kStagedKPairs);
static_assert(!TestPipeline<ProgressiveM128D64Problem>::kStagedKPairs);
static_assert(std::is_same_v<ck_tile::detail::QrTdmPaddingSelection<ProgressiveM64D256Problem>::K,
                             ck_tile::detail::LdsPaddingConfig<false, 0, 0>>);
static_assert(std::is_same_v<ck_tile::detail::QrTdmPaddingSelection<ProgressiveM64D256Problem>::V,
                             ck_tile::detail::LdsPaddingConfig<false, 0, 0>>);
// K-pair staging is a traversal optimization; FP16/BF16 and attention
// features do not change its LDS fragment lifetime contract.
static_assert(TestPipeline<ProgressiveM128Problem>::kStagedKPairs);

template <typename Problem>
using TestGemm0 = ck_tile::remove_cvref_t<
    decltype(ck_tile::BlockFmhaPipelineQRKSVSTdmDefaultPolicy::GetQKBlockGemm<Problem>())>;

static_assert(TestGemm0<ProgressiveM64Problem>::MIterPerWarp == 1);
static_assert(TestGemm0<ProgressiveM128Problem>::MIterPerWarp == 2);
static_assert(ProgressiveM64Problem::BlockFmhaShape::kSubQKHeaddim /
                  ProgressiveM64Problem::BlockFmhaShape::kK0 ==
              4);
static_assert(ProgressiveM128Problem::BlockFmhaShape::kSubQKHeaddim /
                  ProgressiveM128Problem::BlockFmhaShape::kK0 ==
              4);

template <typename Problem>
using TestEpilogue =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename Problem::OaccDataType,
                                                                 typename Problem::ODataType,
                                                                 false,
                                                                 false>>;

template <typename Problem>
using TestKernel = ck_tile::FmhaFwdKernel<TestPipeline<Problem>, TestEpilogue<Problem>>;

constexpr ck_tile::index_t kBatch   = 1;
constexpr ck_tile::index_t kHeads   = 1;
constexpr ck_tile::index_t kSeqlenK = 64;
constexpr ck_tile::index_t kHeadDim = 128;

template <typename DataType = ck_tile::half_t>
std::vector<DataType>
make_input(std::size_t element_count, int multiplier, int modulus, int center, float scale)
{
    std::vector<DataType> values(element_count);
    for(std::size_t i = 0; i < element_count; ++i)
    {
        const int value = (static_cast<int>(i % modulus) * multiplier) % modulus - center;
        values[i]       = ck_tile::type_convert<DataType>(value * scale);
    }
    return values;
}

template <typename Problem>
std::vector<typename Problem::ODataType>
run_kernel(const ck_tile::DeviceMem& q_device,
           const ck_tile::DeviceMem& k_device,
           const ck_tile::DeviceMem& v_device,
           ck_tile::index_t seqlen_q,
           ck_tile::index_t seqlen_k = kSeqlenK,
           ck_tile::GenericAttentionMaskEnum mask_type =
               ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT,
           ck_tile::index_t window_size_left = -1,
           ck_tile::index_t sink_size        = 0,
           const void* sink_ptr              = nullptr)
{
    using Kernel   = TestKernel<Problem>;
    using DataType = typename Problem::ODataType;

    // Derived from the problem, so head dims other than 128 reuse this harness.
    constexpr ck_tile::index_t hdim_q = Problem::BlockFmhaShape::kQKHeaddim;
    constexpr ck_tile::index_t hdim_v = Problem::BlockFmhaShape::kN1;

    std::vector<DataType> output(kBatch * kHeads * seqlen_q * hdim_v);
    ck_tile::DeviceMem output_device(output.size() * sizeof(DataType));
    output_device.SetBytePattern(0x7f);

    typename Kernel::Kargs args{};
    args.q_ptr          = q_device.GetDeviceBuffer();
    args.k_ptr          = k_device.GetDeviceBuffer();
    args.v_ptr          = v_device.GetDeviceBuffer();
    args.o_ptr          = output_device.GetDeviceBuffer();
    args.seqlen_q       = seqlen_q;
    args.seqlen_k       = seqlen_k;
    args.hdim_q         = hdim_q;
    args.hdim_v         = hdim_v;
    args.num_head_q     = kHeads;
    args.nhead_ratio_qk = 1;
    const float scale_s = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hdim_q)));
#if CK_TILE_FMHA_FWD_FAST_EXP2
    args.scale_s = scale_s * ck_tile::log2e_v<>;
#else
    args.scale_s = scale_s;
#endif
    args.stride_q         = hdim_q;
    args.stride_k         = hdim_q;
    args.stride_v         = hdim_v;
    args.stride_o         = hdim_v;
    args.nhead_stride_q   = seqlen_q * hdim_q;
    args.nhead_stride_k   = seqlen_k * hdim_q;
    args.nhead_stride_v   = seqlen_k * hdim_v;
    args.nhead_stride_o   = seqlen_q * hdim_v;
    args.num_head_q_total = kHeads;
    args.batch_stride_q   = kHeads * args.nhead_stride_q;
    args.batch_stride_k   = kHeads * args.nhead_stride_k;
    args.batch_stride_v   = kHeads * args.nhead_stride_v;
    args.batch_stride_o   = kHeads * args.nhead_stride_o;

    if constexpr(Problem::FmhaMask::IsMasking)
    {
        args.window_size_left  = window_size_left;
        args.window_size_right = 0;
        args.mask_type         = mask_type;
        args.sink_size         = sink_size;
    }
    args.sink_ptr = sink_ptr;

    const ck_tile::stream_config stream{};
    ck_tile::launch_kernel(stream,
                           ck_tile::make_kernel<Kernel::kBlockPerCu, ck_tile::gfx125_t>(
                               Kernel{},
                               Kernel::GridSize(kBatch, kHeads, seqlen_q, hdim_v),
                               Kernel::BlockSize(),
                               0,
                               args));
    output_device.FromDevice(output.data());
    return output;
}

template <typename DataType>
void expect_finite_nonzero(const std::vector<DataType>& output)
{
    bool has_nonzero = false;
    for(std::size_t i = 0; i < output.size(); ++i)
    {
        const float value = ck_tile::type_convert<float>(output[i]);
        ASSERT_TRUE(std::isfinite(value)) << "non-finite output at index " << i;
        has_nonzero = has_nonzero || std::abs(value) > 1.0e-3f;
    }
    EXPECT_TRUE(has_nonzero);
}

template <typename DataType>
void expect_close(const std::vector<DataType>& actual,
                  const std::vector<DataType>& expected,
                  const std::string& message)
{
    constexpr double rtol = 1.0e-3;
    constexpr double atol = 1.0e-3;
    EXPECT_TRUE(ck_tile::check_err(actual, expected, message, rtol, atol));
}

TEST(QrTdmProgressiveKLds, M64BufferingModesProduceEquivalentOutput)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    constexpr ck_tile::index_t seqlen_q = 64;
    const auto q = make_input(kBatch * kHeads * seqlen_q * kHeadDim, 13, 29, 14, 0.03125f);
    const auto k = make_input(kBatch * kHeads * kSeqlenK * kHeadDim, 7, 31, 15, 0.025f);
    const auto v = make_input(kBatch * kHeads * kSeqlenK * kHeadDim, 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(ck_tile::half_t));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(ck_tile::half_t));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(ck_tile::half_t));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    const auto single = run_kernel<SingleBufferM64Problem>(q_device, k_device, v_device, seqlen_q);
    const auto baseline =
        run_kernel<DoubleBufferM64Problem>(q_device, k_device, v_device, seqlen_q);
    const auto progressive =
        run_kernel<ProgressiveM64Problem>(q_device, k_device, v_device, seqlen_q);

    expect_finite_nonzero(single);
    expect_finite_nonzero(baseline);
    expect_finite_nonzero(progressive);
    expect_close(baseline, single, "M64 double-buffer baseline differs from single-buffer output");
    expect_close(progressive, single, "M64 progressive output differs from single-buffer output");
}

// The single-buffer main loop is unrolled by two so that each K LDS buffer is
// named at compile time, and the last iteration re-issues its K prefetch onto a
// parked window to keep the number of in-flight TDM loads constant. Sweep one
// to three K tiles, with and without a partial last tile, so both the odd and
// the even tail run.
TEST(QrTdmProgressiveKLds, SingleBufferTileCountParity)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    constexpr ck_tile::index_t seqlen_q     = 64;
    constexpr ck_tile::index_t max_seqlen_k = 192;

    const auto q = make_input(kBatch * kHeads * seqlen_q * kHeadDim, 13, 29, 14, 0.03125f);
    const auto k = make_input(kBatch * kHeads * max_seqlen_k * kHeadDim, 7, 31, 15, 0.025f);
    const auto v = make_input(kBatch * kHeads * max_seqlen_k * kHeadDim, 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(ck_tile::half_t));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(ck_tile::half_t));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(ck_tile::half_t));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    for(const ck_tile::index_t seqlen_k : {64, 65, 128, 129, 192})
    {
        const auto single =
            run_kernel<SingleBufferM64Problem>(q_device, k_device, v_device, seqlen_q, seqlen_k);
        const auto baseline =
            run_kernel<DoubleBufferM64Problem>(q_device, k_device, v_device, seqlen_q, seqlen_k);

        expect_finite_nonzero(single);
        expect_close(single,
                     baseline,
                     "single-buffer output differs from the double-buffer baseline at seqlen_k=" +
                         std::to_string(seqlen_k));
    }
}

// With sink tokens the mainloop first walks [0, sink_seq_end) and then jumps
// the K/V windows to the sliding-window range. K is prefetched one iteration
// ahead of V, so the K window has to jump one iteration earlier than the V and
// bias windows do -- an off-by-one there silently feeds gemm0 the wrong tile.
// Sizes are picked so that a q tile really has both phases: kM0 is 64, so the
// last q tile starts at row 192, a 64-wide left window puts its key range at
// [128, 256), and 64 sink tokens round up to a sink range of [0, 64).
TEST(QrTdmProgressiveKLds, SingleBufferSinkWindowJump)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    constexpr ck_tile::index_t seqlen_q         = 256;
    constexpr ck_tile::index_t seqlen_k         = 256;
    constexpr ck_tile::index_t window_size_left = 63;
    constexpr ck_tile::index_t sink_size        = 64;

    const auto q = make_input(kBatch * kHeads * seqlen_q * kHeadDim, 13, 29, 14, 0.03125f);
    const auto k = make_input(kBatch * kHeads * seqlen_k * kHeadDim, 7, 31, 15, 0.025f);
    const auto v = make_input(kBatch * kHeads * seqlen_k * kHeadDim, 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(ck_tile::half_t));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(ck_tile::half_t));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(ck_tile::half_t));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    const std::vector<float> sink(kHeads, 0.5f);
    const ck_tile::DeviceMem sink_device(sink.size() * sizeof(float));
    sink_device.ToDevice(sink.data());

    const auto single = run_kernel<SingleBufferSinkM64Problem>(
        q_device,
        k_device,
        v_device,
        seqlen_q,
        seqlen_k,
        ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT,
        window_size_left,
        sink_size,
        sink_device.GetDeviceBuffer());
    const auto baseline = run_kernel<DoubleBufferSinkM64Problem>(
        q_device,
        k_device,
        v_device,
        seqlen_q,
        seqlen_k,
        ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT,
        window_size_left,
        sink_size,
        sink_device.GetDeviceBuffer());

    expect_finite_nonzero(single);
    expect_close(
        single, baseline, "sink single-buffer output differs from the double-buffer baseline");
}

// Progressive reload changes when K fragments are pulled out of LDS, not what
// they contain, so the single-buffer path must agree with itself without the
// reload and with the double-buffer baseline. Same odd/even tile counts as
// above, since the reload is issued from inside the unrolled body.
TEST(QrTdmProgressiveKLds, SingleBufferProgressiveMatchesBaseline)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    constexpr ck_tile::index_t seqlen_q     = 64;
    constexpr ck_tile::index_t max_seqlen_k = 192;

    const auto q = make_input(kBatch * kHeads * seqlen_q * kHeadDim, 13, 29, 14, 0.03125f);
    const auto k = make_input(kBatch * kHeads * max_seqlen_k * kHeadDim, 7, 31, 15, 0.025f);
    const auto v = make_input(kBatch * kHeads * max_seqlen_k * kHeadDim, 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(ck_tile::half_t));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(ck_tile::half_t));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(ck_tile::half_t));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    for(const ck_tile::index_t seqlen_k : {64, 65, 128, 129, 192})
    {
        const auto progressive = run_kernel<SingleBufferProgressiveM64Problem>(
            q_device, k_device, v_device, seqlen_q, seqlen_k);
        const auto single =
            run_kernel<SingleBufferM64Problem>(q_device, k_device, v_device, seqlen_q, seqlen_k);
        const auto baseline =
            run_kernel<DoubleBufferM64Problem>(q_device, k_device, v_device, seqlen_q, seqlen_k);

        const std::string suffix = " at seqlen_k=" + std::to_string(seqlen_k);
        expect_finite_nonzero(progressive);
        expect_close(progressive, single, "progressive output differs from single-buffer" + suffix);
        expect_close(progressive,
                     baseline,
                     "progressive output differs from double-buffer baseline" + suffix);
    }
}

// Cross-check the progressive reload against the plain single-buffer path at a
// head dim other than 128, over the caller's K tile counts.
template <typename BaseProblem, typename ProgressiveProblem>
void expect_progressive_matches_single_buffer(const std::string& label,
                                              const std::vector<ck_tile::index_t>& seqlen_ks)
{
    static_assert(std::is_same_v<typename BaseProblem::BlockFmhaShape,
                                 typename ProgressiveProblem::BlockFmhaShape>);
    static_assert(!BaseProblem::kUseDoubleKVLdsBuffer);
    static_assert(!BaseProblem::kProgressiveDsLoadK);
    static_assert(!ProgressiveProblem::kUseDoubleKVLdsBuffer);
    static_assert(ProgressiveProblem::kProgressiveDsLoadK);

    using DataType                      = typename BaseProblem::QDataType;
    constexpr ck_tile::index_t hdim     = BaseProblem::BlockFmhaShape::kQKHeaddim;
    constexpr ck_tile::index_t seqlen_q = 64;

    ck_tile::index_t max_seqlen_k = 0;
    for(const ck_tile::index_t seqlen_k : seqlen_ks)
        max_seqlen_k = ck_tile::max(max_seqlen_k, seqlen_k);

    const auto q = make_input<DataType>(kBatch * kHeads * seqlen_q * hdim, 13, 29, 14, 0.03125f);
    const auto k = make_input<DataType>(
        static_cast<std::size_t>(kBatch * kHeads * max_seqlen_k * hdim), 7, 31, 15, 0.025f);
    const auto v = make_input<DataType>(
        static_cast<std::size_t>(kBatch * kHeads * max_seqlen_k * hdim), 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(DataType));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(DataType));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(DataType));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    for(const ck_tile::index_t seqlen_k : seqlen_ks)
    {
        const auto baseline =
            run_kernel<BaseProblem>(q_device, k_device, v_device, seqlen_q, seqlen_k);
        const auto progressive =
            run_kernel<ProgressiveProblem>(q_device, k_device, v_device, seqlen_q, seqlen_k);

        expect_finite_nonzero(progressive);
        expect_close(progressive,
                     baseline,
                     label + " progressive output differs from the single-buffer baseline at" +
                         " seqlen_k=" + std::to_string(seqlen_k));
    }
}

// Three and four K tiles put the x2-unrolled mainloop through a full steady
// state -- parity 0, parity 1, then wrapping back to parity 0 -- instead of
// stopping inside the first unrolled pair the way 64/65 do.
TEST(QrTdmProgressiveKLds, D256SingleBufferProgressiveMatchesBaseline)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    expect_progressive_matches_single_buffer<SingleBufferD256Problem,
                                             SingleBufferProgressiveD256Problem>(
        "d256", {64, 65, 192, 193});
}

// Kept to one and two K tiles on purpose: d512 needs 196 KiB of LDS and runs a
// single workgroup per CU.
TEST(QrTdmProgressiveKLds, D512SingleBufferProgressiveMatchesBaseline)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    expect_progressive_matches_single_buffer<SingleBufferD512Problem,
                                             SingleBufferProgressiveD512Problem>("d512", {64, 65});
}

template <typename DataType>
class QrTdmProgressiveKLdsM128 : public ::testing::Test
{
};

using M128DataTypes = ::testing::Types<ck_tile::half_t, ck_tile::bf16_t>;
TYPED_TEST_SUITE(QrTdmProgressiveKLdsM128, M128DataTypes);

TYPED_TEST(QrTdmProgressiveKLdsM128, SingleKBlockProducesEquivalentOutput)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    constexpr ck_tile::index_t seqlen_q = 128;
    const auto q =
        make_input<TypeParam>(kBatch * kHeads * seqlen_q * kHeadDim, 13, 29, 14, 0.03125f);
    const auto k = make_input<TypeParam>(kBatch * kHeads * kSeqlenK * kHeadDim, 7, 31, 15, 0.025f);
    const auto v = make_input<TypeParam>(kBatch * kHeads * kSeqlenK * kHeadDim, 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(TypeParam));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(TypeParam));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(TypeParam));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    const auto baseline = run_kernel<TestFmhaProblem<128, true, false, TypeParam>>(
        q_device, k_device, v_device, seqlen_q);
    const auto progressive = run_kernel<TestFmhaProblem<128, true, true, TypeParam>>(
        q_device, k_device, v_device, seqlen_q);

    expect_finite_nonzero(baseline);
    expect_finite_nonzero(progressive);
    expect_close(progressive, baseline, "M128 progressive output differs from baseline output");
}

TYPED_TEST(QrTdmProgressiveKLdsM128, MultipleKBlocksProduceEquivalentOutput)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    constexpr ck_tile::index_t seqlen_q = 128;
    constexpr ck_tile::index_t seqlen_k = 128;
    const auto q =
        make_input<TypeParam>(kBatch * kHeads * seqlen_q * kHeadDim, 13, 29, 14, 0.03125f);
    const auto k = make_input<TypeParam>(kBatch * kHeads * seqlen_k * kHeadDim, 7, 31, 15, 0.025f);
    const auto v = make_input<TypeParam>(kBatch * kHeads * seqlen_k * kHeadDim, 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(TypeParam));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(TypeParam));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(TypeParam));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    const auto baseline = run_kernel<TestFmhaProblem<128, true, false, TypeParam>>(
        q_device, k_device, v_device, seqlen_q, seqlen_k);
    const auto progressive = run_kernel<TestFmhaProblem<128, true, true, TypeParam>>(
        q_device, k_device, v_device, seqlen_q, seqlen_k);

    expect_finite_nonzero(baseline);
    expect_finite_nonzero(progressive);
    expect_close(
        progressive, baseline, "M128 multi-block progressive output differs from baseline");
}

TYPED_TEST(QrTdmProgressiveKLdsM128, CausalKPairStagingPreservesOutputAcrossTileCounts)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    using Mask = ck_tile::GenericAttentionMask<true, false>;
    // Different query tiles consume different numbers of KV blocks, including
    // the short causal prefix and the final ping-pong buffer reuse. A staged
    // pair overwritten before its last consumer must disagree with the
    // non-progressive full-K reference below. FP16 exercises the same staged
    // traversal rather than a data-type-selected fallback.
    for(const ck_tile::index_t seqlen : {128, 256, 384})
    {
        SCOPED_TRACE(seqlen);
        const auto q =
            make_input<TypeParam>(kBatch * kHeads * seqlen * kHeadDim, 13, 29, 14, 0.03125f);
        const auto k =
            make_input<TypeParam>(kBatch * kHeads * seqlen * kHeadDim, 7, 31, 15, 0.025f);
        const auto v =
            make_input<TypeParam>(kBatch * kHeads * seqlen * kHeadDim, 11, 37, 18, 0.02f);
        const ck_tile::DeviceMem q_device(q.size() * sizeof(TypeParam));
        const ck_tile::DeviceMem k_device(k.size() * sizeof(TypeParam));
        const ck_tile::DeviceMem v_device(v.size() * sizeof(TypeParam));
        q_device.ToDevice(q.data());
        k_device.ToDevice(k.data());
        v_device.ToDevice(v.data());

        const auto baseline = run_kernel<TestFmhaProblem<128, true, false, TypeParam, Mask>>(
            q_device, k_device, v_device, seqlen, seqlen);
        const auto progressive = run_kernel<TestFmhaProblem<128, true, true, TypeParam, Mask>>(
            q_device, k_device, v_device, seqlen, seqlen);

        expect_finite_nonzero(baseline);
        expect_finite_nonzero(progressive);
        expect_close(progressive, baseline, "M128 causal K staging differs from full-K output");
    }
}

template <typename DataType, typename Mask>
void check_long_sequence_sum(ck_tile::index_t seqlen, float input_scale)
{
    SCOPED_TRACE(Mask::IsMasking ? "causal" : "dense");
    // Reference retains the original serial row sum through the non-progressive
    // path. Long sequences exercise repeated l updates, not just one KV tile.
    const auto count = kBatch * kHeads * seqlen * kHeadDim;
    const auto q     = make_input<DataType>(count, 13, 29, 14, input_scale);
    auto k           = make_input<DataType>(count, 7, 31, 15, input_scale);
    // Periodic K alone reaches its dense row maxima in the first tile. Grow
    // subsequent blocks so this also exercises non-unit online rescaling.
    for(std::size_t i = 0; i < k.size(); ++i)
    {
        const auto block   = (i / kHeadDim) / 64;
        const float growth = 1.0f + static_cast<float>(block) / 64.0f;
        k[i] = ck_tile::type_convert<DataType>(ck_tile::type_convert<float>(k[i]) * growth);
    }
    // Positive V avoids periodic cancellation toward zero on long sequences;
    // a wrong denominator must remain visible at the existing output tolerance.
    const auto v = make_input<DataType>(count, 11, 37, -7, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(DataType));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(DataType));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(DataType));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());
    const auto baseline = run_kernel<TestFmhaProblem<128, true, false, DataType, Mask>>(
        q_device, k_device, v_device, seqlen, seqlen);
    const auto candidate = run_kernel<TestFmhaProblem<128, true, true, DataType, Mask>>(
        q_device, k_device, v_device, seqlen, seqlen);
    expect_finite_nonzero(baseline);
    expect_finite_nonzero(candidate);
    expect_close(candidate, baseline, "long-sequence denominator differs from serial-sum path");
}

TYPED_TEST(QrTdmProgressiveKLdsM128, LongSequencesPreserveDenominatorAccuracy)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";
    for(const ck_tile::index_t seqlen : {4096, 8192, 16384, 32768})
    {
        SCOPED_TRACE(seqlen);
        for(const float input_scale : {0.03125f, 0.25f})
        {
            SCOPED_TRACE(input_scale);
            check_long_sequence_sum<TypeParam, ck_tile::SimplifiedGenericAttentionMask<false>>(
                seqlen, input_scale);
            check_long_sequence_sum<TypeParam, ck_tile::GenericAttentionMask<true, false>>(
                seqlen, input_scale);
        }
    }
}

template <typename DataType, typename Mask>
void check_mixed_row_rescaling()
{
    SCOPED_TRACE(Mask::IsMasking ? "causal" : "dense");
    constexpr ck_tile::index_t seqlen = 256;
    constexpr float levels[4][2]      = {{4, 4}, {4, 8}, {8, 8}, {8, 16}};
    constexpr float values[4]         = {1, 0.5f, 0.25f, 0.125f};
    std::vector<DataType> q(seqlen * kHeadDim, ck_tile::type_convert<DataType>(0));
    std::vector<DataType> k(seqlen * kHeadDim, ck_tile::type_convert<DataType>(0));
    std::vector<DataType> v(seqlen * kHeadDim);
    for(ck_tile::index_t row = 0; row < seqlen; ++row)
    {
        // Even/odd query rows alternate unchanged/changing maxima in each
        // tile. A first-lane-only skip decision or omitted O correction must
        // disagree with the always-rescaling reference, within the same wave.
        q[row * kHeadDim + row % 2] = ck_tile::type_convert<DataType>(4);
        for(ck_tile::index_t col = 0; col < 2; ++col)
            k[row * kHeadDim + col] = ck_tile::type_convert<DataType>(levels[row / 64][col]);
        for(ck_tile::index_t col = 0; col < kHeadDim; ++col)
            v[row * kHeadDim + col] =
                ck_tile::type_convert<DataType>(values[row / 64] + col / 256.0f);
    }
    const ck_tile::DeviceMem q_device(q.size() * sizeof(DataType));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(DataType));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(DataType));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());
    const auto baseline = run_kernel<TestFmhaProblem<128, true, false, DataType, Mask>>(
        q_device, k_device, v_device, seqlen, seqlen);
    const auto candidate = run_kernel<TestFmhaProblem<128, true, true, DataType, Mask>>(
        q_device, k_device, v_device, seqlen, seqlen);
    expect_finite_nonzero(baseline);
    expect_finite_nonzero(candidate);
    expect_close(candidate, baseline, "mixed row maxima lost an O rescaling update");
}

TYPED_TEST(QrTdmProgressiveKLdsM128, MixedRowMaximumChangesPreserveRescaling)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";
    check_mixed_row_rescaling<TypeParam, ck_tile::SimplifiedGenericAttentionMask<false>>();
    check_mixed_row_rescaling<TypeParam, ck_tile::GenericAttentionMask<true, false>>();
}

template <typename DataType, typename Mask>
void compare_cross_attention(ck_tile::index_t seqlen_q,
                             ck_tile::index_t seqlen_k,
                             ck_tile::GenericAttentionMaskEnum mask_type,
                             ck_tile::index_t empty_rows = 0)
{
    SCOPED_TRACE(seqlen_q);
    SCOPED_TRACE(seqlen_k);
    const auto q = make_input<DataType>(seqlen_q * kHeadDim, 13, 29, 14, 0.03125f);
    const auto k = make_input<DataType>(seqlen_k * kHeadDim, 7, 31, 15, 0.025f);
    const auto v = make_input<DataType>(seqlen_k * kHeadDim, 11, 37, 18, 0.02f);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(DataType));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(DataType));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(DataType));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());

    const auto baseline = run_kernel<TestFmhaProblem<128, true, false, DataType, Mask>>(
        q_device, k_device, v_device, seqlen_q, seqlen_k, mask_type);
    const auto progressive = run_kernel<TestFmhaProblem<128, true, true, DataType, Mask>>(
        q_device, k_device, v_device, seqlen_q, seqlen_k, mask_type);
    expect_finite_nonzero(baseline);
    expect_finite_nonzero(progressive);
    expect_close(progressive, baseline, "M128 cross-attention differs from full-K output");
    for(ck_tile::index_t i = 0; i < empty_rows * kHeadDim; ++i)
        EXPECT_EQ(ck_tile::type_convert<float>(progressive[i]), 0.0f);
}

TYPED_TEST(QrTdmProgressiveKLdsM128, GenericDenseMaskPreservesQueryTail)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    compare_cross_attention<TypeParam, ck_tile::GenericAttentionMask<false, false>>(
        129, 192, ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
}

// A custom range can have a logical tail within a fully allocated K/V tile.
// IsMasking=false alone must not disable the pipeline's score-tail clamp.
struct PrefixDenseMask : ck_tile::SimplifiedGenericAttentionMask<false>
{
    using ck_tile::SimplifiedGenericAttentionMask<false>::SimplifiedGenericAttentionMask;

    template <ck_tile::index_t YTile, ck_tile::index_t XTile>
    CK_TILE_HOST_DEVICE constexpr auto GetTileRangeAlongX(ck_tile::index_t row,
                                                          ck_tile::number<YTile> height,
                                                          ck_tile::number<XTile> width) const
    {
        const auto range =
            ck_tile::SimplifiedGenericAttentionMask<false>::GetTileRangeAlongX(row, height, width);
        return ck_tile::make_tuple(0, range.at(ck_tile::number<1>{}) - 1);
    }
};

TYPED_TEST(QrTdmProgressiveKLdsM128, CustomDenseSubrangeRetainsScoreTailMask)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    constexpr ck_tile::index_t seqlen_q = 129;
    constexpr ck_tile::index_t seqlen_k = 192;
    const std::vector<TypeParam> q(seqlen_q * kHeadDim, ck_tile::type_convert<TypeParam>(0));
    const std::vector<TypeParam> k(seqlen_k * kHeadDim, ck_tile::type_convert<TypeParam>(0));
    std::vector<TypeParam> v(seqlen_k * kHeadDim, ck_tile::type_convert<TypeParam>(1));
    for(ck_tile::index_t col = 0; col < kHeadDim; ++col)
        v[(seqlen_k - 1) * kHeadDim + col] = ck_tile::type_convert<TypeParam>(16);
    const ck_tile::DeviceMem q_device(q.size() * sizeof(TypeParam));
    const ck_tile::DeviceMem k_device(k.size() * sizeof(TypeParam));
    const ck_tile::DeviceMem v_device(v.size() * sizeof(TypeParam));
    q_device.ToDevice(q.data());
    k_device.ToDevice(k.data());
    v_device.ToDevice(v.data());
    const auto output = run_kernel<TestFmhaProblem<128, true, true, TypeParam, PrefixDenseMask>>(
        q_device, k_device, v_device, seqlen_q, seqlen_k);
    // All 191 included rows have V=1; including the excluded V=16 gives 1.078125.
    for(std::size_t i = 0; i < output.size(); ++i)
        ASSERT_NEAR(ck_tile::type_convert<float>(output[i]), 1.0f, 1.0e-3f) << i;
}

TYPED_TEST(QrTdmProgressiveKLdsM128, QueryTailsPreserveOutputWithThreeKBlocks)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    // Wrong K/V bounds or tail rotation can corrupt the last valid query rows.
    // This output check alone cannot detect reads of unused out-of-bounds Q rows.
    for(const ck_tile::index_t seqlen_q : {1, 127, 129, 193})
    {
        compare_cross_attention<TypeParam, ck_tile::SimplifiedGenericAttentionMask<false>>(
            seqlen_q, 192, ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
        compare_cross_attention<TypeParam, ck_tile::GenericAttentionMask<true, false>>(
            seqlen_q, 192, ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
    }
}

TYPED_TEST(QrTdmProgressiveKLdsM128, BottomRightCausalPreservesEmptyAndOddTilePrefixes)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "QR-TDM progressive K LDS is only supported on gfx1250";

    // Bottom-right alignment gives Q-K empty rows: 64 and 320 respectively.
    // The first case reaches one- and three-KV-tile prefixes; the second also
    // exercises entire query tiles that must return before issuing any TDM.
    using Mask = ck_tile::GenericAttentionMask<true, false>;
    compare_cross_attention<TypeParam, Mask>(
        256, 192, ck_tile::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT, 64);
    compare_cross_attention<TypeParam, Mask>(
        384, 64, ck_tile::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT, 320);
}

} // namespace qr_tdm_progressive_k_lds_test
