// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 29: FMHA Backward with ALiBi Bias + Dropout
//
// Demonstrates:
//   1. Forward kernel with alibi bias + dropout + LSE
//   2. Backward kernel families with alibi + dropout
//   3. GPU forward execution with alibi bias, validates output
//   4. Backward plan with all features enabled
//   5. How deterministic mode affects the backward plan
//
// Backward kernels use planning only -- actual backward GPU execution requires
// all 3 stages to compile, and bwd_dq_dk_dv has tile structure issues on gfx950.

#include <hip/hip_runtime.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;

DECL_FMHA_KERNEL_SET(bwd_bias_dropout_fmha_kernels,
                     // Forward: alibi bias + dropout + LSE
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("no")
                              .bias("alibi")
                              .lse(true)
                              .dropout(true)
                              .qscale("no"),
                          FmhaAlgorithm()
                              .tile_m0(128)
                              .tile_n0(128)
                              .tile_k0(32)
                              .tile_n1(128)
                              .tile_k1(32)
                              .tile_k0max(128)
                              .wave_m0(4)
                              .wave_n0(1)
                              .wave_k0(1)
                              .wave_m1(4)
                              .wave_n1(1)
                              .wave_k1(1)
                              .warp_m0(32)
                              .warp_n0(32)
                              .warp_k0(16)
                              .warp_m1(32)
                              .warp_n1(32)
                              .warp_k1(16)
                              .pipeline("qr_async")
                              .padding(true, true, true, true)
                              .alignments(128, 128)
                              .selection_rank(0),
                          "gfx950")

                         // Backward stage 1: dot(dO, O) with alibi + dropout (non-deterministic)
                         .add(FmhaSignature()
                                  .family("bwd_dot_do_o")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("alibi")
                                  .dropout(true)
                                  .dbias(false)
                                  .store_randval(false)
                                  .deterministic(false),
                              FmhaAlgorithm()
                                  .tile_m0(64)
                                  .tile_n0(128)
                                  .tile_k0(32)
                                  .tile_n1(0)
                                  .tile_k1(0)
                                  .tile_k0max(0)
                                  .padding(true, true, true, true)
                                  .selection_rank(0),
                              "gfx950")

                         // Backward stage 2: dQ, dK, dV with alibi + dropout (non-deterministic)
                         .add(FmhaSignature()
                                  .family("bwd_dq_dk_dv")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("alibi")
                                  .dropout(true)
                                  .dbias(false)
                                  .store_randval(false)
                                  .deterministic(false),
                              FmhaAlgorithm()
                                  .tile_m0(16)
                                  .tile_n0(128)
                                  .tile_k0(128)
                                  .tile_n1(16)
                                  .tile_k1(128)
                                  .tile_k0max(32)
                                  .wave(1, 4, 1, 4, 1, 1, 1, 4, 1)
                                  .warp(16, 16, 32, 16, 16, 16, 16, 16, 16)
                                  .padding(true, true, true, true)
                                  .max_seq_len_q(0)
                                  .selection_rank(0),
                              "gfx950")

                         // Backward stage 3: convert dQ with alibi + dropout (non-deterministic)
                         .add(FmhaSignature()
                                  .family("bwd_convert_dq")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("alibi")
                                  .dropout(true)
                                  .dbias(false)
                                  .store_randval(false)
                                  .deterministic(false),
                              FmhaAlgorithm()
                                  .tile_m0(64)
                                  .tile_n0(128)
                                  .tile_k0(0)
                                  .tile_n1(0)
                                  .tile_k1(0)
                                  .tile_k0max(0)
                                  .padding(true, true, true, true)
                                  .selection_rank(0),
                              "gfx950")

                         // Deterministic variants for comparison
                         .add(FmhaSignature()
                                  .family("bwd_dot_do_o")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("alibi")
                                  .dropout(true)
                                  .dbias(false)
                                  .store_randval(false)
                                  .deterministic(true),
                              FmhaAlgorithm()
                                  .tile_m0(64)
                                  .tile_n0(128)
                                  .tile_k0(32)
                                  .tile_n1(0)
                                  .tile_k1(0)
                                  .tile_k0max(0)
                                  .padding(true, true, true, true)
                                  .selection_rank(0),
                              "gfx950")

                         .add(FmhaSignature()
                                  .family("bwd_dq_dk_dv")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("alibi")
                                  .dropout(true)
                                  .dbias(false)
                                  .store_randval(false)
                                  .deterministic(true),
                              FmhaAlgorithm()
                                  .tile_m0(16)
                                  .tile_n0(128)
                                  .tile_k0(128)
                                  .tile_n1(16)
                                  .tile_k1(128)
                                  .tile_k0max(32)
                                  .wave(1, 4, 1, 4, 1, 1, 1, 4, 1)
                                  .warp(16, 16, 32, 16, 16, 16, 16, 16, 16)
                                  .padding(true, true, true, true)
                                  .max_seq_len_q(0)
                                  .selection_rank(0),
                              "gfx950")

                         .add(FmhaSignature()
                                  .family("bwd_convert_dq")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("alibi")
                                  .dropout(true)
                                  .dbias(false)
                                  .store_randval(false)
                                  .deterministic(true),
                              FmhaAlgorithm()
                                  .tile_m0(64)
                                  .tile_n0(128)
                                  .tile_k0(0)
                                  .tile_n1(0)
                                  .tile_k1(0)
                                  .tile_k0max(0)
                                  .padding(true, true, true, true)
                                  .selection_rank(0),
                              "gfx950"));

namespace {

using FmhaDataType = ck_tile::fp16_t;

void cpu_attention_fwd_alibi(const std::vector<float>& Q,
                             const std::vector<float>& K,
                             const std::vector<float>& V,
                             std::vector<float>& O,
                             std::vector<float>& LSE,
                             int batch,
                             int nhead,
                             int seqlen,
                             int hdim,
                             float scale,
                             const std::vector<float>& alibi_slopes)
{
    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < nhead; ++h)
        {
            const float slope = alibi_slopes[h];

            for(int sq = 0; sq < seqlen; ++sq)
            {
                std::vector<float> scores(seqlen, 0.0f);
                float max_score = -1e30f;

                for(int sk = 0; sk < seqlen; ++sk)
                {
                    float dot = 0.0f;
                    for(int d = 0; d < hdim; ++d)
                    {
                        int q_idx = ((b * nhead + h) * seqlen + sq) * hdim + d;
                        int k_idx = ((b * nhead + h) * seqlen + sk) * hdim + d;
                        dot += Q[q_idx] * K[k_idx];
                    }
                    scores[sk] = dot * scale + slope * static_cast<float>(sk - sq);
                    max_score  = std::max(max_score, scores[sk]);
                }

                float sum_exp = 0.0f;
                for(int sk = 0; sk < seqlen; ++sk)
                {
                    scores[sk] = std::exp(scores[sk] - max_score);
                    sum_exp += scores[sk];
                }

                int lse_idx  = (b * nhead + h) * seqlen + sq;
                LSE[lse_idx] = max_score + std::log(sum_exp);

                for(int sk = 0; sk < seqlen; ++sk)
                    scores[sk] /= sum_exp;

                for(int dv = 0; dv < hdim; ++dv)
                {
                    float acc = 0.0f;
                    for(int sk = 0; sk < seqlen; ++sk)
                    {
                        int v_idx = ((b * nhead + h) * seqlen + sk) * hdim + dv;
                        acc += scores[sk] * V[v_idx];
                    }
                    int o_idx = ((b * nhead + h) * seqlen + sq) * hdim + dv;
                    O[o_idx]  = acc;
                }
            }
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 29: FMHA Backward with Bias + Dropout",
                     "ALiBi bias + dropout forward (GPU) + backward plan with deterministic mode");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--nhead", "4", "Number of heads");
    args.add_option("--seqlen", "64", "Sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 2);
    const int nhead            = args.get_int("--nhead", 4);
    const int seqlen           = args.get_int("--seqlen", 64);
    const int hdim             = args.get_int("--hdim", 128);
    const float scale          = 1.0f / std::sqrt(static_cast<float>(hdim));

    print_header("Example 29: FMHA Backward with ALiBi Bias + Dropout");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("bwd_bias_dropout_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 3);

    // Step 2: Plan backward (non-deterministic) with alibi + dropout
    std::cout << "\nStep 2: Plan Backward (non-deterministic, alibi + dropout)\n";

    fmha_bwd_traits bwd_traits{};
    bwd_traits.hdim_q           = hdim;
    bwd_traits.hdim_v           = hdim;
    bwd_traits.data_type        = "fp16";
    bwd_traits.is_group_mode    = false;
    bwd_traits.mask_type        = mask_enum::no_mask;
    bwd_traits.bias_type        = bias_enum::alibi;
    bwd_traits.has_dbias        = false;
    bwd_traits.has_dropout      = true;
    bwd_traits.is_store_randval = false;
    bwd_traits.is_deterministic = false;

    fmha_bwd_args bwd_args{};
    bwd_args.batch        = batch;
    bwd_args.seqlen_q     = seqlen;
    bwd_args.seqlen_k     = seqlen;
    bwd_args.max_seqlen_q = seqlen;
    bwd_args.max_seqlen_k = seqlen;
    bwd_args.hdim_q       = hdim;
    bwd_args.hdim_v       = hdim;
    bwd_args.nhead_q      = nhead;
    bwd_args.nhead_k      = nhead;

    auto nondet_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(bwd_traits, bwd_args), gfx_arch));

    if(nondet_plan.is_valid() && nondet_plan.stages.size() >= 2)
    {
        std::cout << "  Non-deterministic plan stages (" << nondet_plan.stages.size() << "):\n";
        for(const auto& stage : nondet_plan.stages)
        {
            std::cout << "    " << to_string(stage.family) << " -> " << stage.kernel_id << "\n";
        }
    }
    else
    {
        std::cout << "  Non-deterministic plan: INVALID or single-stage\n";
    }

    // Step 2b: Plan backward (deterministic) with alibi + dropout
    std::cout << "\nStep 2b: Plan Backward (deterministic, alibi + dropout)\n";

    fmha_bwd_traits det_traits  = bwd_traits;
    det_traits.is_deterministic = true;

    auto det_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(det_traits, bwd_args), gfx_arch));

    if(det_plan.is_valid() && det_plan.stages.size() >= 2)
    {
        std::cout << "  Deterministic plan stages (" << det_plan.stages.size() << "):\n";
        for(const auto& stage : det_plan.stages)
        {
            std::cout << "    " << to_string(stage.family) << " -> " << stage.kernel_id << "\n";
        }
    }
    else
    {
        std::cout << "  Deterministic plan: INVALID or single-stage\n";
    }

    std::cout << "\n  Deterministic mode difference:\n";
    std::cout << "    Non-det: dQ accumulated via atomic adds (faster, non-reproducible)\n";
    std::cout << "    Det:     dQ accumulated with split-stride (slower, bit-reproducible)\n";

    // Step 3: Run forward on GPU with alibi bias + dropout
    std::cout << "\nStep 3: Run Forward (alibi + dropout, GPU)\n";

    const int64_t qkv_elems     = static_cast<int64_t>(batch) * nhead * seqlen * hdim;
    const int64_t lse_elems     = static_cast<int64_t>(batch) * nhead * seqlen;
    const int64_t randval_elems = static_cast<int64_t>(batch) * nhead * seqlen * seqlen;

    GpuBuffer<FmhaDataType> q_dev(qkv_elems);
    GpuBuffer<FmhaDataType> k_dev(qkv_elems);
    GpuBuffer<FmhaDataType> v_dev(qkv_elems);
    GpuBuffer<FmhaDataType> o_dev(qkv_elems);
    GpuBuffer<float> lse_dev(lse_elems);
    GpuBuffer<uint8_t> rand_val_dev(randval_elems);

    // ALiBi slopes: geometric series
    std::vector<float> alibi_slopes_host(nhead);
    for(int h = 0; h < nhead; ++h)
        alibi_slopes_host[h] = -std::pow(2.0f, -(8.0f * (h + 1) / nhead));

    GpuBuffer<float> alibi_slopes_dev(nhead);
    alibi_slopes_dev.copy_from_host(alibi_slopes_host.data());

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<FmhaDataType> q_host(qkv_elems), k_host(qkv_elems), v_host(qkv_elems);
    for(auto& x : q_host)
        x = FmhaDataType(dist(rng));
    for(auto& x : k_host)
        x = FmhaDataType(dist(rng));
    for(auto& x : v_host)
        x = FmhaDataType(dist(rng));

    q_dev.copy_from_host(q_host.data());
    k_dev.copy_from_host(k_host.data());
    v_dev.copy_from_host(v_host.data());
    o_dev.zero();
    lse_dev.zero();
    rand_val_dev.zero();

    std::cout << "  ALiBi slopes: [";
    for(int h = 0; h < nhead; ++h)
    {
        if(h > 0)
            std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << alibi_slopes_host[h];
    }
    std::cout << "]\n";

    fmha_fwd_traits fwd_traits{};
    fwd_traits.hdim_q              = hdim;
    fwd_traits.hdim_v              = hdim;
    fwd_traits.data_type           = "fp16";
    fwd_traits.is_group_mode       = false;
    fwd_traits.is_v_rowmajor       = true;
    fwd_traits.has_logits_soft_cap = false;
    fwd_traits.mask_type           = mask_enum::no_mask;
    fwd_traits.bias_type           = bias_enum::alibi;
    fwd_traits.has_lse             = true;
    fwd_traits.has_dropout         = true;
    fwd_traits.qscale_type         = quant_scale_enum::no_scale;

    fmha_fwd_args fwd_args{};
    fwd_args.q_ptr   = q_dev.get();
    fwd_args.k_ptr   = k_dev.get();
    fwd_args.v_ptr   = v_dev.get();
    fwd_args.o_ptr   = o_dev.get();
    fwd_args.lse_ptr = lse_dev.get();

    fwd_args.bias_ptr                   = alibi_slopes_dev.get();
    fwd_args.rand_val_ptr               = rand_val_dev.get();
    fwd_args.q_descale_ptr              = nullptr;
    fwd_args.k_descale_ptr              = nullptr;
    fwd_args.v_descale_ptr              = nullptr;
    fwd_args.sink_ptr                   = nullptr;
    fwd_args.block_scale_seqstart_q_ptr = nullptr;
    fwd_args.block_scale_seqstart_k_ptr = nullptr;

    fwd_args.seqlen_q        = seqlen;
    fwd_args.seqlen_k        = seqlen;
    fwd_args.batch           = batch;
    fwd_args.max_seqlen_q    = seqlen;
    fwd_args.hdim_q          = hdim;
    fwd_args.hdim_v          = hdim;
    fwd_args.nhead_q         = nhead;
    fwd_args.nhead_k         = nhead;
    fwd_args.scale_s         = scale;
    fwd_args.logits_soft_cap = 0.0f;

    fwd_args.stride_q       = hdim;
    fwd_args.stride_k       = hdim;
    fwd_args.stride_v       = hdim;
    fwd_args.stride_bias    = 0; // alibi: per-head slope, no spatial stride
    fwd_args.stride_randval = seqlen;
    fwd_args.stride_o       = hdim;

    fwd_args.nhead_stride_q         = seqlen * hdim;
    fwd_args.nhead_stride_k         = seqlen * hdim;
    fwd_args.nhead_stride_v         = seqlen * hdim;
    fwd_args.nhead_stride_bias      = 1; // alibi: stride between slopes
    fwd_args.nhead_stride_randval   = seqlen * seqlen;
    fwd_args.nhead_stride_lse       = seqlen;
    fwd_args.nhead_stride_o         = seqlen * hdim;
    fwd_args.nhead_stride_q_descale = 0;
    fwd_args.nhead_stride_k_descale = 0;
    fwd_args.nhead_stride_v_descale = 0;

    fwd_args.batch_stride_q         = nhead * seqlen * hdim;
    fwd_args.batch_stride_k         = nhead * seqlen * hdim;
    fwd_args.batch_stride_v         = nhead * seqlen * hdim;
    fwd_args.batch_stride_bias      = 0; // alibi slopes shared across batch
    fwd_args.batch_stride_randval   = nhead * seqlen * seqlen;
    fwd_args.batch_stride_lse       = nhead * seqlen;
    fwd_args.batch_stride_o         = nhead * seqlen * hdim;
    fwd_args.batch_stride_q_descale = 0;
    fwd_args.batch_stride_k_descale = 0;
    fwd_args.batch_stride_v_descale = 0;

    fwd_args.window_size_left    = -1;
    fwd_args.window_size_right   = -1;
    fwd_args.sink_size           = 0;
    fwd_args.mask_type           = 0;
    fwd_args.min_seqlen_q        = 0;
    fwd_args.p_drop              = 0.2f;
    fwd_args.s_randval           = true;
    fwd_args.drop_seed_offset    = std::make_pair(uint64_t(42), uint64_t(0));
    fwd_args.block_scale_size_q  = 0;
    fwd_args.block_scale_size_kv = 0;

    bool fwd_passed = false;
    try
    {
        float fwd_time = dispatcher.run_fwd(fwd_traits, fwd_args, nullptr);
        std::cout << "  Forward time: " << std::fixed << std::setprecision(4) << fwd_time
                  << " ms\n";

        auto problem =
            FmhaProblem::from_invocation(FmhaInvocation::make(fwd_traits, fwd_args), gfx_arch);
        double tflops = static_cast<double>(problem.num_ops()) / (fwd_time * 1e-3) / 1e12;
        std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";
        fwd_passed = true;
    }
    catch(const std::exception& e)
    {
        std::cerr << "  Forward ERROR: " << e.what() << "\n";
    }

    // Step 4: Validate forward output (without dropout reference -- just check non-zero + LSE)
    std::cout << "\nStep 4: Validate Forward Output\n";

    if(fwd_passed)
    {
        std::vector<FmhaDataType> o_host(qkv_elems);
        o_dev.copy_to_host(o_host.data());

        int nonzero = 0;
        for(int64_t i = 0; i < qkv_elems; ++i)
        {
            if(static_cast<float>(o_host[i]) != 0.0f)
                ++nonzero;
        }
        std::cout << "  Non-zero outputs: " << nonzero << " / " << qkv_elems << "\n";

        std::vector<float> lse_host(lse_elems);
        lse_dev.copy_to_host(lse_host.data());

        int lse_reasonable = 0;
        for(int64_t i = 0; i < lse_elems; ++i)
        {
            if(std::isfinite(lse_host[i]) && std::abs(lse_host[i]) < 100.0f)
                ++lse_reasonable;
        }
        std::cout << "  LSE reasonable: " << lse_reasonable << " / " << lse_elems << "\n";

        std::cout << "  LSE sample [0..3]: ";
        for(int i = 0; i < std::min<int>(4, lse_elems); ++i)
            std::cout << std::fixed << std::setprecision(4) << lse_host[i] << " ";
        std::cout << "\n";

        fwd_passed = (nonzero > 0) && (lse_reasonable == lse_elems);

        // ALiBi reference (without dropout) for sanity check on bias effect
        std::vector<float> q_f32(qkv_elems), k_f32(qkv_elems), v_f32(qkv_elems);
        for(int64_t i = 0; i < qkv_elems; ++i)
            q_f32[i] = static_cast<float>(q_host[i]);
        for(int64_t i = 0; i < qkv_elems; ++i)
            k_f32[i] = static_cast<float>(k_host[i]);
        for(int64_t i = 0; i < qkv_elems; ++i)
            v_f32[i] = static_cast<float>(v_host[i]);

        std::vector<float> o_ref(qkv_elems, 0.0f);
        std::vector<float> lse_ref(lse_elems, 0.0f);
        cpu_attention_fwd_alibi(q_f32,
                                k_f32,
                                v_f32,
                                o_ref,
                                lse_ref,
                                batch,
                                nhead,
                                seqlen,
                                hdim,
                                scale,
                                alibi_slopes_host);

        // LSE should be close (dropout doesn't change LSE in the CK implementation --
        // LSE is computed before dropout is applied to the attention weights)
        double max_lse_err = 0.0;
        for(int64_t i = 0; i < lse_elems; ++i)
            max_lse_err =
                std::max(max_lse_err, static_cast<double>(std::abs(lse_host[i] - lse_ref[i])));

        std::cout << "  LSE vs alibi ref (no dropout) max error: " << std::scientific << max_lse_err
                  << "\n";
    }

    // Step 5: Show backward API pattern with all features
    std::cout << "\nStep 5: Backward API Pattern (all features)\n";
    std::cout << "  bwd_traits.bias_type        = alibi\n";
    std::cout << "  bwd_traits.has_dropout      = true\n";
    std::cout << "  bwd_traits.is_store_randval = false\n";
    std::cout << "  bwd_traits.has_dbias        = false (alibi has no learnable params)\n";
    std::cout << "\n  Non-deterministic plan: " << nondet_plan.stages.size() << " stage(s)\n";
    std::cout << "  Deterministic plan:     " << det_plan.stages.size() << " stage(s)\n";
    std::cout << "\n  Key backward args for dropout:\n";
    std::cout << "    bwd_args.p_drop         = 0.2\n";
    std::cout << "    bwd_args.p_undrop       = 1.0 / (1.0 - p_drop) = 1.25\n";
    std::cout << "    bwd_args.drop_seed_offset = {42, 0} (must match fwd)\n";

    print_separator();
    std::cout << "Status: " << (fwd_passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return fwd_passed ? 0 : 1;
}
