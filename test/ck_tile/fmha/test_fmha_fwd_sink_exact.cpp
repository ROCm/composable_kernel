// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Pin represented inputs to exercise coherent P-rounding error independently of random fills.
// With V=1, O is the real-token probability mass and LSE is computed from the same logits.

#include "ck_tile/host.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "gtest/gtest.h"

#include <cmath>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

// Built only into the fp8-input binaries that keep a wider output (see CMakeLists). An fp8
// ODataType is excluded on purpose: e4m3 carries 3 mantissa bits, so the output cast alone
// costs up to ~6% and would swamp the ~2% gain this test exists to catch.
using QDataType = FmhaFwdTypeConfig<DataTypeConfig>::QDataType;
using ODataType = FmhaFwdTypeConfig<DataTypeConfig>::ODataType;

constexpr const char* prec_string()
{
    if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp8Fp32>)
        return "fp8fp32";
    else
        return "fp8bf16";
}

struct SinkExactResult
{
    double max_relative_error = 0; // vs exact attention, per output element
    double max_lse_error      = 0; // vs exact log-sum-exp, per row
    bool all_finite           = true;
    bool dispatched           = false;
};

SinkExactResult run_sink_exact(
    int hdim, int seqlen_k, int seqlen_q, float sink, bool store_lse, bool varying = false)
{
    SinkExactResult result;

    const bool has_sink = std::isfinite(sink);
    const float one     = 1.f;

    std::vector<QDataType> q(seqlen_q * hdim, ck_tile::type_convert<QDataType>(0.f));
    std::vector<QDataType> k(seqlen_k * hdim, ck_tile::type_convert<QDataType>(0.f));
    std::vector<QDataType> v(seqlen_k * hdim, ck_tile::type_convert<QDataType>(1.f));
    std::vector<ODataType> o(seqlen_q * hdim, ck_tile::type_convert<ODataType>(NAN));
    std::vector<float> lse(seqlen_q, NAN);
    if(varying)
    {
        for(int i = 0; i < seqlen_q; ++i)
            q[i * hdim] = ck_tile::type_convert<QDataType>(1.f);
        for(int j = 0; j < seqlen_k; ++j)
            k[j * hdim] = ck_tile::type_convert<QDataType>(-8.f + (j % 9 - 4) * 0.25f);
    }

    ck_tile::DeviceMem q_buf(q.size()), k_buf(k.size()), v_buf(v.size());
    ck_tile::DeviceMem o_buf(o.size() * sizeof(ODataType)), lse_buf(lse.size() * sizeof(float));
    ck_tile::DeviceMem sink_buf(sizeof(float)), scale_buf(sizeof(float));

    q_buf.ToDevice(q.data());
    k_buf.ToDevice(k.data());
    v_buf.ToDevice(v.data());
    o_buf.ToDevice(o.data());
    lse_buf.ToDevice(lse.data());
    sink_buf.ToDevice(&sink);
    scale_buf.ToDevice(&one);

    fmha_fwd_args args{};
    args.q_ptr    = q_buf.GetDeviceBuffer();
    args.k_ptr    = k_buf.GetDeviceBuffer();
    args.v_ptr    = v_buf.GetDeviceBuffer();
    args.o_ptr    = o_buf.GetDeviceBuffer();
    args.lse_ptr  = store_lse ? lse_buf.GetDeviceBuffer() : nullptr;
    args.sink_ptr = has_sink ? sink_buf.GetDeviceBuffer() : nullptr;
    // PERTENSOR with unit descales: the represented values above are the real values.
    args.q_descale_ptr = args.k_descale_ptr = args.v_descale_ptr = scale_buf.GetDeviceBuffer();

    args.batch = args.nhead_q = args.nhead_k = args.num_head_q_total = 1;
    args.seqlen_q = args.max_seqlen_q = seqlen_q;
    args.seqlen_k                     = seqlen_k;
    args.hdim_q = args.hdim_v = hdim;
    args.scale_s              = varying ? 1.f : 1.f / std::sqrt(static_cast<float>(hdim));

    args.stride_q = args.stride_k = args.stride_v = args.stride_o = hdim;
    args.nhead_stride_q = args.batch_stride_q = seqlen_q * hdim;
    args.nhead_stride_o = args.batch_stride_o = seqlen_q * hdim;
    args.nhead_stride_k = args.batch_stride_k = seqlen_k * hdim;
    args.nhead_stride_v = args.batch_stride_v = seqlen_k * hdim;
    args.nhead_stride_lse = args.batch_stride_lse = seqlen_q;
    args.window_size_left = args.window_size_right = -1;
    args.block_scale_size_q = args.block_scale_size_kv = 128;

    fmha_fwd_traits traits{hdim,
                           hdim,
                           prec_string(),
                           false, // batch mode
                           true,  // v row-major
                           false, // no logits soft cap
                           mask_enum::no_mask,
                           bias_enum::no_bias,
                           store_lse,
                           false, // no dropout
                           quant_scale_enum::pertensor,
                           false, // no skip_min_seqlen_q
                           has_sink};

    if(fmha_fwd(traits, args, ck_tile::stream_config{nullptr, false}) < 0)
        return result; // dispatched stays false: no instance for this trait set
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    result.dispatched = true;

    o_buf.FromDevice(o.data());
    if(store_lse)
        lse_buf.FromDevice(lse.data());

    double real_mass = 0;
    for(int j = 0; j < seqlen_k; ++j)
    {
        double score = 0;
        for(int c = 0; c < hdim; ++c)
            score += static_cast<double>(ck_tile::type_convert<float>(q[c])) *
                     ck_tile::type_convert<float>(k[j * hdim + c]);
        real_mass += std::exp(score * args.scale_s);
    }
    const double denominator = real_mass + (has_sink ? std::exp(static_cast<double>(sink)) : 0.0);
    const double reference   = real_mass / denominator;

    for(const auto& element : o)
    {
        const double value = ck_tile::type_convert<float>(element);
        result.all_finite &= std::isfinite(value);
        result.max_relative_error =
            std::max(result.max_relative_error, std::abs(value / reference - 1));
    }
    if(store_lse)
    {
        const double reference_lse = std::log(denominator);
        for(float value : lse)
        {
            result.all_finite &= std::isfinite(value);
            result.max_lse_error = std::max(result.max_lse_error,
                                            std::abs(static_cast<double>(value) - reference_lse));
        }
    }
    return result;
}

// hdim, seqlen_k, seqlen_q, sink, store_lse
class SinkExact : public ::testing::TestWithParam<std::tuple<int, int, int, float, bool>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    SinkExact,
    ::testing::Combine(::testing::Values(64, 128),  // both fp8 qr_tdm head dims
                       ::testing::Values(128, 500), // multi-tile, and a non-multiple tail
                       ::testing::Values(63, 64),   // padded and exact query tiles
                       ::testing::Values(0.5f,      // sink above the zero-valued real logits
                                         4.f,       // sink dominates the row max
                                         6.f,       // sink dominates by a wide margin
                                         -ck_tile::numeric<float>::infinity()), // no-sink control
                       ::testing::Bool()));

TEST_P(SinkExact, MatchesExactAttention)
{
    const auto [hdim, seqlen_k, seqlen_q, sink, store_lse] = GetParam();

    const auto result = run_sink_exact(hdim, seqlen_k, seqlen_q, sink, store_lse);
    if(!result.dispatched)
        GTEST_SKIP() << "No instance for current parameters";

    EXPECT_TRUE(result.all_finite);
    // Same 1% output gain bound the quantization-scale suites use.
    EXPECT_LT(result.max_relative_error, 0.01);
    // LSE is the mathematical log-sum-exp: it carries the unquantized row sum, so it is
    // held to fp32 accumulation error rather than to the fp8 output bound.
    if(store_lse)
        EXPECT_LT(result.max_lse_error, 1e-4);
}

TEST(SinkQuantizedMass, NonuniformRealScores)
{
    for(int hdim : {64, 128})
    {
        for(float sink : {-4.f, -ck_tile::numeric<float>::infinity()})
        {
            SCOPED_TRACE(::testing::Message() << "hdim=" << hdim << " sink=" << sink);
            const auto result = run_sink_exact(hdim, 500, 63, sink, true, true);
            if(!result.dispatched)
                GTEST_SKIP() << "No instance for current parameters";
            EXPECT_TRUE(result.all_finite);
            EXPECT_LT(result.max_relative_error, 0.01);
            EXPECT_LT(result.max_lse_error, 1e-4);
        }
    }
}

} // namespace
