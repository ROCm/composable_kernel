// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Direct scheduled-kernel regression tests with an independent FP32 oracle.
// Sequence-padded batch specializations exercise the sink mechanism; generated
// dispatcher selection and unpadded D192 batch variants need separate coverage.
#include "fmha_test_common.hpp"
#include "fmha_reference.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CK_TILE_FMHA_SINK_TEST_HEAD_DIM
#define CK_TILE_FMHA_SINK_TEST_HEAD_DIM 128
#endif
#ifndef CK_TILE_FMHA_SINK_TEST_TILE_N
#define CK_TILE_FMHA_SINK_TEST_TILE_N 128
#endif
#ifdef CK_TILE_FMHA_SINK_TEST_FP16
using Data                   = ck_tile::half_t;
constexpr const char* kDtype = "fp16";
#else
using Data                   = ck_tile::bf16_t;
constexpr const char* kDtype = "bf16";
#endif
constexpr int kD  = CK_TILE_FMHA_SINK_TEST_HEAD_DIM;
constexpr int kN  = CK_TILE_FMHA_SINK_TEST_TILE_N;
constexpr int kDv = 128;

void Check(hipError_t rc)
{
    if(rc != hipSuccess)
        throw std::runtime_error(hipGetErrorString(rc));
}

template <typename T>
struct Buffer
{
    T* ptr = nullptr;
    explicit Buffer(std::size_t count)
    {
        Check(hipMalloc(&ptr, std::max(count, std::size_t{1}) * sizeof(T)));
    }
    ~Buffer()
    {
        if(ptr)
            (void)hipFree(ptr);
    }
    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;
    void Upload(const std::vector<T>& values)
    {
        if(!values.empty())
            Check(hipMemcpy(ptr, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    void Download(std::vector<T>& values)
    {
        if(!values.empty())
            Check(hipMemcpy(values.data(), ptr, values.size() * sizeof(T), hipMemcpyDeviceToHost));
    }
};

struct Config
{
    int sq = 512, sk = 512, left = 31, right = 0, sink = 1;
    int b = 2, h = 4, hk = 2;
    bool masked = true, has_sink = true, bottom_right = false, bhsd = false, lse = true;
    bool sink_logit = false;
    float scale     = 0.19f;
};

std::size_t Offset(int b, int h, int s, int d, int heads, int rows, int dim, bool bhsd)
{
    return bhsd ? ((static_cast<std::size_t>(b) * heads + h) * rows + s) * dim + d
                : ((static_cast<std::size_t>(b) * rows + s) * heads + h) * dim + d;
}

// Independent window semantics. The prefix still obeys the right boundary;
// it extends only the left boundary. No CK mask helper participates in the oracle.
bool Valid(const Config& c, int q, int k)
{
    if(k < 0 || k >= c.sk)
        return false;
    if(!c.masked)
        return true;
    const int center       = q + (c.bottom_right ? c.sk - c.sq : 0);
    const bool below_right = c.right < 0 || k <= center + c.right;
    const bool above_left  = c.left < 0 || k >= center - c.left;
    return below_right && (above_left || (c.has_sink && k < c.sink));
}

template <bool Mask, bool HasSink, bool Lse>
void Launch(const Config& c, fmha_fwd_args args)
{
    using Base   = typename ck_tile::test::Model<kD, false, Mask, Lse, Data, kN>::Problem;
    using Traits = ck_tile::TileFmhaTraits<true,
                                           true,
                                           true,
                                           true,
                                           false,
                                           ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                           false,
                                           Lse,
                                           false,
                                           ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                           -1,
                                           false,
                                           HasSink>;
    using Problem =
        ck_tile::BlockFmhaPipelineProblem<Data,
                                          Data,
                                          Data,
                                          float,
                                          float,
                                          Data,
                                          std::uint8_t,
                                          float,
                                          Data,
                                          float,
                                          Data,
                                          typename Base::BlockFmhaShape,
                                          false,
                                          ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>,
                                          ck_tile::SimplifiedGenericAttentionMask<Mask>,
                                          false,
                                          Traits>;
    using Policy   = ck_tile::FmhaTdmSchedPolicyFor<Problem>;
    using Pipeline = ck_tile::BlockFmhaPipelineQRKSVSTdmSched<Problem, Policy>;
    using Epilogue =
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, Data, true, true>>;
    using Kernel = ck_tile::FmhaFwdKernel<Pipeline, Epilogue>;
    static_assert(Kernel::kHasSink == HasSink && Kernel::kStoreLSE == Lse);
    auto [kargs, grids]         = fmha_fwd_create_kargs_and_grids<Kernel>(args);
    using NoVirtualSinkPipeline = ck_tile::BlockFmhaPipelineQRKSVSTdmSched<Problem, Policy, false>;
    using NoVirtualSinkKernel   = ck_tile::FmhaFwdKernel<NoVirtualSinkPipeline, Epilogue>;
    auto [no_virtual_sink_kargs, no_virtual_sink_grids] =
        fmha_fwd_create_kargs_and_grids<NoVirtualSinkKernel>(args);
    EXPECT_EQ(NoVirtualSinkKernel::IsSupportedArgument(no_virtual_sink_kargs),
              args.sink_ptr == nullptr);
    const bool finite_virtual_sink = args.sink_ptr != nullptr;
    std::cout << "sink_kernel=qr_tdm_sched dtype=" << kDtype << " dq=" << kD << " n=" << kN
              << " mask=" << Mask << " token_sink=" << HasSink << " lse=" << Lse
              << " virtual_sink=" << finite_virtual_sink << " occupancy=" << Kernel::kBlockPerCu
              << " grid=" << grids.x << ',' << grids.y << ',' << grids.z
              << " folded_scale=" << kargs.scale_s << '\n';
    const ck_tile::stream_config stream{nullptr, false, 0, 0, 1};
    if(finite_virtual_sink)
    {
        ck_tile::launch_kernel(stream,
                               ck_tile::make_kernel<Kernel::kBlockPerCu, ck_tile::gfx125_t>(
                                   Kernel{}, grids, Kernel::BlockSize(), 0, kargs));
    }
    else
    {
        ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel<NoVirtualSinkKernel::kBlockPerCu, ck_tile::gfx125_t>(
                NoVirtualSinkKernel{},
                no_virtual_sink_grids,
                NoVirtualSinkKernel::BlockSize(),
                0,
                no_virtual_sink_kargs));
    }
    Check(hipGetLastError());
    Check(hipDeviceSynchronize());
    (void)c;
}

template <bool Mask, bool HasSink>
void SelectLse(const Config& c, const fmha_fwd_args& args)
{
    if(c.lse)
        Launch<Mask, HasSink, true>(c, args);
    else
        Launch<Mask, HasSink, false>(c, args);
}

int CheckSink(const Config& c)
{
    int device = 0;
    hipDeviceProp_t prop{};
    Check(hipGetDevice(&device));
    Check(hipGetDeviceProperties(&prop, device));
    std::cout << std::setprecision(10) << "sink_device visible=" << device
              << " arch=" << prop.gcnArchName << " cus=" << prop.multiProcessorCount << '\n';
    auto make_input = [&](int heads, int rows, int dim, int phase) {
        std::vector<Data> values(static_cast<std::size_t>(c.b) * heads * rows * dim);
        for(int b = 0; b < c.b; ++b)
            for(int h = 0; h < heads; ++h)
                for(int s = 0; s < rows; ++s)
                    for(int d = 0; d < dim; ++d)
                    {
                        const float value =
                            0.35f * std::sin((s + 1) * (0.061f + phase * 0.017f) +
                                             (d + 1) * 0.137f + h * 0.53f + b * 0.29f) +
                            0.11f * std::cos((s + 1) * (d % 7 + 1) * 0.031f + phase * 0.4f);
                        values[Offset(b, h, s, d, heads, rows, dim, c.bhsd)] =
                            ck_tile::type_convert<Data>(value);
                    }
        return values;
    };
    auto q = make_input(c.h, c.sq, kD, 0);
    auto k = make_input(c.hk, c.sk, kD, 1);
    auto v = make_input(c.hk, c.sk, kDv, 2);
    std::vector<Data> o(static_cast<std::size_t>(c.b) * c.h * c.sq * kDv,
                        ck_tile::type_convert<Data>(std::numeric_limits<float>::quiet_NaN()));
    std::vector<float> lse(static_cast<std::size_t>(c.b) * c.h * c.sq,
                           std::numeric_limits<float>::quiet_NaN());
    std::vector<float> sinks{-0.7f, 0.3f, 3.2f, 5.1f};
    Buffer<Data> qb(q.size()), kb(k.size()), vb(v.size()), ob(o.size());
    Buffer<float> lb(lse.size()), sb(sinks.size());
    qb.Upload(q);
    kb.Upload(k);
    vb.Upload(v);
    ob.Upload(o);
    lb.Upload(lse);
    sb.Upload(sinks);
    fmha_fwd_args args{};
    args.q_ptr             = qb.ptr;
    args.k_ptr             = kb.ptr;
    args.v_ptr             = vb.ptr;
    args.o_ptr             = ob.ptr;
    args.lse_ptr           = c.lse ? lb.ptr : nullptr;
    args.sink_ptr          = c.sink_logit ? sb.ptr : nullptr;
    args.seqlen_q          = c.sq;
    args.seqlen_k          = c.sk;
    args.max_seqlen_q      = c.sq;
    args.hdim_q            = kD;
    args.hdim_v            = kDv;
    args.batch             = c.b;
    args.nhead_q           = c.h;
    args.nhead_k           = c.hk;
    args.scale_s           = c.scale;
    args.stride_q          = c.bhsd ? kD : c.h * kD;
    args.stride_k          = c.bhsd ? kD : c.hk * kD;
    args.stride_v          = c.bhsd ? kDv : c.hk * kDv;
    args.stride_o          = c.bhsd ? kDv : c.h * kDv;
    args.nhead_stride_q    = c.bhsd ? c.sq * kD : kD;
    args.nhead_stride_k    = c.bhsd ? c.sk * kD : kD;
    args.nhead_stride_v    = c.bhsd ? c.sk * kDv : kDv;
    args.nhead_stride_o    = c.bhsd ? c.sq * kDv : kDv;
    args.batch_stride_q    = c.h * c.sq * kD;
    args.batch_stride_k    = c.hk * c.sk * kD;
    args.batch_stride_v    = c.hk * c.sk * kDv;
    args.batch_stride_o    = c.h * c.sq * kDv;
    args.nhead_stride_lse  = c.sq;
    args.batch_stride_lse  = c.h * c.sq;
    args.window_size_left  = c.left;
    args.window_size_right = c.right;
    args.sink_size         = c.sink;
    args.mask_type =
        static_cast<int>(c.bottom_right ? ck_tile::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT
                                        : ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
    if(!c.masked)
        SelectLse<false, false>(c, args);
    else if(c.has_sink)
        SelectLse<true, true>(c, args);
    else
        SelectLse<true, false>(c, args);
    ob.Download(o);
    if(c.lse)
        lb.Download(lse);
    namespace reference = ck_tile::test::reference;
    auto decode         = [](const std::vector<Data>& input) {
        std::vector<float> output(input.size());
        std::transform(input.begin(), input.end(), output.begin(), [](Data value) {
            return ck_tile::type_convert<float>(value);
        });
        return output;
    };
    const auto q_float = decode(q), k_float = decode(k), v_float = decode(v), o_float = decode(o);
    auto view = [&](const auto& values, int batch, int heads, int rows, int dimensions) {
        return reference::FloatView{
            values.data(),
            values.size(),
            static_cast<std::size_t>(batch) * heads * rows * dimensions,
            static_cast<std::size_t>(c.bhsd ? rows * dimensions : dimensions),
            static_cast<std::size_t>(c.bhsd ? dimensions : heads * dimensions),
            1};
    };
    std::vector<reference::Sequence> sequences;
    std::vector<reference::Output> outputs;
    for(int batch = 0; batch < c.b; ++batch)
    {
        reference::Sequence input;
        input.q               = view(q_float, batch, c.h, c.sq, kD);
        input.k               = view(k_float, batch, c.hk, c.sk, kD);
        input.v               = view(v_float, batch, c.hk, c.sk, kDv);
        input.query_length    = c.sq;
        input.key_length      = c.sk;
        input.query_heads     = c.h;
        input.kv_heads        = c.hk;
        input.query_dimension = kD;
        input.value_dimension = kDv;
        input.score_scale     = c.scale;
        input.is_masked       = [c](std::size_t query, std::size_t key) {
            return !Valid(c, static_cast<int>(query), static_cast<int>(key));
        };
        if(c.sink_logit)
            input.sink_logits.assign(sinks.begin(), sinks.begin() + c.h);
        sequences.push_back(std::move(input));
        reference::Output output{view(o_float, batch, c.h, c.sq, kDv), std::nullopt};
        if(c.lse)
            output.lse = reference::FloatView{lse.data(),
                                              lse.size(),
                                              static_cast<std::size_t>(batch) * c.h * c.sq,
                                              static_cast<std::size_t>(c.sq),
                                              1,
                                              1};
        outputs.push_back(output);
    }
    const auto expected = reference::Compute(sequences);
    const reference::AccuracyContract contract{std::is_same_v<Data, ck_tile::half_t> ? 10 : 7, kN};
    const auto metrics = reference::Compare(sequences, expected, outputs, {}, contract);
    const bool pass    = metrics.passed;
    std::cout << "sink_metrics passed=" << pass << " checked_rows=" << metrics.checked_rows
              << " checked_o=" << metrics.checked_elements
              << " checked_lse=" << metrics.checked_lse_rows << " max_o=" << metrics.max_abs_error
              << " max_lse=" << metrics.max_lse_abs_error << " min_cos=" << metrics.min_row_cosine
              << " relative_l2=" << metrics.relative_rms_error
              << " max_component_ratio=" << metrics.max_component_ratio
              << " bad_components=" << metrics.component_mismatch_count
              << " strict_passed=" << metrics.strict_passed
              << " contract_passed=" << metrics.contract_passed << '\n';
    return pass ? 0 : 1;
}

namespace {
struct SinkCase
{
    const char* name;
    Config config;
};
const std::array<SinkCase, 8> kSinkCases{{
    {"DenseNoVirtual",
     Config{.sq       = 128,
            .sk       = 128,
            .left     = -1,
            .right    = -1,
            .sink     = 0,
            .masked   = false,
            .has_sink = false}},
    {"CausalNoVirtual", Config{.sq = 128, .sk = 128, .left = -1, .sink = 0, .has_sink = false}},
    {"DenseVirtual",
     Config{.sq         = 128,
            .sk         = 128,
            .left       = -1,
            .right      = -1,
            .sink       = 0,
            .masked     = false,
            .has_sink   = false,
            .sink_logit = true}},
    {"CausalVirtual",
     Config{.sq = 128, .sk = 128, .left = -1, .sink = 0, .has_sink = false, .sink_logit = true}},
    {"LocalPrefix", Config{}},
    {"LocalPrefixAcrossTiles", Config{.sq = 768, .sk = 768, .sink = kN + 1}},
    {"LocalPrefixAndVirtual", Config{.sink_logit = true}},
    {"EmptyKeysVirtual",
     Config{.sq         = 128,
            .sk         = 0,
            .left       = -1,
            .right      = -1,
            .sink       = 0,
            .masked     = false,
            .has_sink   = false,
            .sink_logit = true}},
}};
class QrTdmSchedSink : public ck_tile::test::Gfx125FmhaTest,
                       public ::testing::WithParamInterface<std::tuple<int, bool>>
{
};
TEST_P(QrTdmSchedSink, DirectPaddedBatch)
{
    const auto [case_index, lse] = GetParam();
    auto config                  = kSinkCases[case_index].config;
    config.lse                   = lse;
    SCOPED_TRACE(kSinkCases[case_index].name);
    EXPECT_EQ(CheckSink(config), 0);
}
INSTANTIATE_TEST_SUITE_P(VirtualAndRealPrefix,
                         QrTdmSchedSink,
                         ::testing::Combine(::testing::Range(0, 8), ::testing::Bool()),
                         [](const ::testing::TestParamInfo<QrTdmSchedSink::ParamType>& case_info) {
                             const auto index = std::get<0>(case_info.param);
                             const auto lse   = std::get<1>(case_info.param);
                             return std::string{kSinkCases[index].name} + (lse ? "Lse" : "NoLse");
                         });
} // namespace
