// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 28: FMHA Backward with Causal Mask
//
// Demonstrates:
//   1. Forward kernel with top_left causal mask + LSE
//   2. Backward kernel families (bwd_dot_do_o, bwd_dq_dk_dv, bwd_convert_dq) with causal mask
//   3. GPU forward execution with causal mask validation
//   4. Backward 3-stage plan display
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

DECL_FMHA_KERNEL_SET(bwd_masks_fmha_kernels,
                     // Forward: causal mask (top_left) with LSE for backward
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("top_left")
                              .bias("no")
                              .lse(true)
                              .dropout(false)
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

                         // Backward stage 1: dot(dO, O) with causal mask
                         .add(FmhaSignature()
                                  .family("bwd_dot_do_o")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("top_left")
                                  .bias("no")
                                  .dropout(false)
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

                         // Backward stage 2: compute dQ, dK, dV with causal mask
                         .add(FmhaSignature()
                                  .family("bwd_dq_dk_dv")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("top_left")
                                  .bias("no")
                                  .dropout(false)
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

                         // Backward stage 3: convert accumulated dQ from fp32 to fp16
                         .add(FmhaSignature()
                                  .family("bwd_convert_dq")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("top_left")
                                  .bias("no")
                                  .dropout(false)
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
                              "gfx950"));

namespace {

using FmhaDataType = ck_tile::fp16_t;

void cpu_attention_fwd_causal(const std::vector<float>& Q,
                              const std::vector<float>& K,
                              const std::vector<float>& V,
                              std::vector<float>& O,
                              std::vector<float>& LSE,
                              int batch,
                              int nhead,
                              int seqlen,
                              int hdim,
                              float scale)
{
    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < nhead; ++h)
        {
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
                    float s = dot * scale;

                    // top_left causal: mask if sk > sq
                    if(sk > sq)
                        s = -1e30f;

                    scores[sk] = s;
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
    ExampleArgs args("Example 28: FMHA Backward with Masks",
                     "Causal mask forward (GPU) + backward plan");
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

    print_header("Example 28: FMHA Backward with Causal Mask");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("bwd_masks_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 3);

    // Step 2: Plan backward (3-stage) with causal mask
    std::cout << "\nStep 2: Plan Backward (causal mask)\n";

    fmha_bwd_traits bwd_traits{};
    bwd_traits.hdim_q           = hdim;
    bwd_traits.hdim_v           = hdim;
    bwd_traits.data_type        = "fp16";
    bwd_traits.is_group_mode    = false;
    bwd_traits.mask_type        = mask_enum::mask_top_left;
    bwd_traits.bias_type        = bias_enum::no_bias;
    bwd_traits.has_dbias        = false;
    bwd_traits.has_dropout      = false;
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

    auto bwd_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(bwd_traits, bwd_args), gfx_arch));

    if(bwd_plan.is_valid() && bwd_plan.stages.size() >= 2)
    {
        std::cout << "  Backward plan stages (" << bwd_plan.stages.size() << "):\n";
        for(const auto& stage : bwd_plan.stages)
        {
            std::cout << "    " << to_string(stage.family) << " -> " << stage.kernel_id << "\n";
        }
    }
    else
    {
        std::cout << "  Backward plan: INVALID or single-stage (expected 3 stages)\n";
        std::cout << "  This is expected -- backward planning shows the pattern\n";
    }

    // Step 3: Run forward on GPU with causal mask
    std::cout << "\nStep 3: Run Forward (causal mask, GPU)\n";

    const int64_t qkv_elems = static_cast<int64_t>(batch) * nhead * seqlen * hdim;
    const int64_t lse_elems = static_cast<int64_t>(batch) * nhead * seqlen;

    GpuBuffer<FmhaDataType> q_dev(qkv_elems);
    GpuBuffer<FmhaDataType> k_dev(qkv_elems);
    GpuBuffer<FmhaDataType> v_dev(qkv_elems);
    GpuBuffer<FmhaDataType> o_dev(qkv_elems);
    GpuBuffer<float> lse_dev(lse_elems);

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

    fmha_fwd_traits fwd_traits{};
    fwd_traits.hdim_q              = hdim;
    fwd_traits.hdim_v              = hdim;
    fwd_traits.data_type           = "fp16";
    fwd_traits.is_group_mode       = false;
    fwd_traits.is_v_rowmajor       = true;
    fwd_traits.has_logits_soft_cap = false;
    fwd_traits.mask_type           = mask_enum::mask_top_left;
    fwd_traits.bias_type           = bias_enum::no_bias;
    fwd_traits.has_lse             = true;
    fwd_traits.has_dropout         = false;
    fwd_traits.qscale_type         = quant_scale_enum::no_scale;

    fmha_fwd_args fwd_args{};
    fwd_args.q_ptr   = q_dev.get();
    fwd_args.k_ptr   = k_dev.get();
    fwd_args.v_ptr   = v_dev.get();
    fwd_args.o_ptr   = o_dev.get();
    fwd_args.lse_ptr = lse_dev.get();

    fwd_args.bias_ptr                   = nullptr;
    fwd_args.q_descale_ptr              = nullptr;
    fwd_args.k_descale_ptr              = nullptr;
    fwd_args.v_descale_ptr              = nullptr;
    fwd_args.rand_val_ptr               = nullptr;
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
    fwd_args.stride_bias    = 0;
    fwd_args.stride_randval = 0;
    fwd_args.stride_o       = hdim;

    fwd_args.nhead_stride_q         = seqlen * hdim;
    fwd_args.nhead_stride_k         = seqlen * hdim;
    fwd_args.nhead_stride_v         = seqlen * hdim;
    fwd_args.nhead_stride_bias      = 0;
    fwd_args.nhead_stride_randval   = 0;
    fwd_args.nhead_stride_lse       = seqlen;
    fwd_args.nhead_stride_o         = seqlen * hdim;
    fwd_args.nhead_stride_q_descale = 0;
    fwd_args.nhead_stride_k_descale = 0;
    fwd_args.nhead_stride_v_descale = 0;

    fwd_args.batch_stride_q         = nhead * seqlen * hdim;
    fwd_args.batch_stride_k         = nhead * seqlen * hdim;
    fwd_args.batch_stride_v         = nhead * seqlen * hdim;
    fwd_args.batch_stride_bias      = 0;
    fwd_args.batch_stride_randval   = 0;
    fwd_args.batch_stride_lse       = nhead * seqlen;
    fwd_args.batch_stride_o         = nhead * seqlen * hdim;
    fwd_args.batch_stride_q_descale = 0;
    fwd_args.batch_stride_k_descale = 0;
    fwd_args.batch_stride_v_descale = 0;

    fwd_args.window_size_left    = -1;
    fwd_args.window_size_right   = 0;
    fwd_args.sink_size           = 0;
    fwd_args.mask_type           = 1; // top_left
    fwd_args.min_seqlen_q        = 0;
    fwd_args.p_drop              = 0.0f;
    fwd_args.s_randval           = false;
    fwd_args.drop_seed_offset    = std::make_pair(uint64_t(0), uint64_t(0));
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

    // Step 4: Validate forward output
    std::cout << "\nStep 4: Validate Forward Output\n";

    if(fwd_passed)
    {
        std::vector<FmhaDataType> o_host(qkv_elems);
        o_dev.copy_to_host(o_host.data());

        std::vector<float> lse_host(lse_elems);
        lse_dev.copy_to_host(lse_host.data());

        std::vector<float> q_f32(qkv_elems), k_f32(qkv_elems), v_f32(qkv_elems);
        for(int64_t i = 0; i < qkv_elems; ++i)
            q_f32[i] = static_cast<float>(q_host[i]);
        for(int64_t i = 0; i < qkv_elems; ++i)
            k_f32[i] = static_cast<float>(k_host[i]);
        for(int64_t i = 0; i < qkv_elems; ++i)
            v_f32[i] = static_cast<float>(v_host[i]);

        std::vector<float> o_ref(qkv_elems, 0.0f);
        std::vector<float> lse_ref(lse_elems, 0.0f);
        cpu_attention_fwd_causal(
            q_f32, k_f32, v_f32, o_ref, lse_ref, batch, nhead, seqlen, hdim, scale);

        double max_o_err  = 0.0;
        int o_errors      = 0;
        const double rtol = 1e-2;
        const double atol = 1e-2;

        for(int64_t i = 0; i < qkv_elems; ++i)
        {
            float gpu_val  = static_cast<float>(o_host[i]);
            float ref_val  = o_ref[i];
            double abs_err = std::abs(gpu_val - ref_val);
            max_o_err      = std::max(max_o_err, abs_err);
            if(abs_err > atol + rtol * std::abs(ref_val))
                ++o_errors;
        }

        double max_lse_err = 0.0;
        int lse_reasonable = 0;
        for(int64_t i = 0; i < lse_elems; ++i)
        {
            if(std::isfinite(lse_host[i]) && std::abs(lse_host[i]) < 100.0f)
                ++lse_reasonable;
            max_lse_err =
                std::max(max_lse_err, static_cast<double>(std::abs(lse_host[i] - lse_ref[i])));
        }

        std::cout << "  Output max abs error: " << std::scientific << max_o_err << "\n";
        std::cout << "  Output errors: " << o_errors << " / " << qkv_elems << "\n";
        std::cout << "  LSE reasonable: " << lse_reasonable << " / " << lse_elems << "\n";
        std::cout << "  LSE max error: " << std::scientific << max_lse_err << "\n";

        fwd_passed = (o_errors == 0) && (lse_reasonable == lse_elems);
    }

    // Step 5: Show backward API pattern
    std::cout << "\nStep 5: Backward API Pattern (traits + args)\n";
    std::cout << "  bwd_traits.mask_type        = mask_top_left\n";
    std::cout << "  bwd_traits.bias_type        = no_bias\n";
    std::cout << "  bwd_traits.has_dropout      = false\n";
    std::cout << "  bwd_traits.is_deterministic = false\n";
    std::cout << "  bwd_args.window_size_left   = -1\n";
    std::cout << "  bwd_args.window_size_right  = 0 (causal)\n";
    std::cout << "  bwd_args.mask_type          = 1 (top_left)\n";
    std::cout << "  Backward plan resolves to " << bwd_plan.stages.size() << " stage(s)\n";

    print_separator();
    std::cout << "Status: " << (fwd_passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return fwd_passed ? 0 : 1;
}
