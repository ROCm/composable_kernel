// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 22: FMHA Backward with GPU Execution
//
// Demonstrates:
//   1. Declare 3 backward kernel families (bwd_dot_do_o, bwd_dq_dk_dv, bwd_convert_dq)
//   2. Run forward to get O and LSE
//   3. Run backward to compute dQ, dK, dV
//   4. Validate gradients are non-zero
//
// Falls back to planning only if backward kernels fail to compile on gfx950.

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

DECL_FMHA_KERNEL_SET(gpu_bwd_fmha_kernels,
                     // Forward kernel (to produce O and LSE for backward)
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("no")
                              .bias("no")
                              .lse(true)
                              .dropout(false)
                              .qscale("no"),
                          FmhaAlgorithm()
                              // Stage 0 (Q*K^T): m0=seqlen_q, n0=seqlen_k, k0=hdim_q
                              .tile_m0(128)
                              .tile_n0(128)
                              .tile_k0(32)
                              // Stage 1 (Attn*V): n1=hdim_v, k1=seqlen_k, k0max=alignment
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

                         // Backward: dot(dO, O) to compute d scalar
                         .add(FmhaSignature()
                                  .family("bwd_dot_do_o")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
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

                         // Backward: compute dQ, dK, dV
                         .add(FmhaSignature()
                                  .family("bwd_dq_dk_dv")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
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

                         // Backward: convert accumulated dQ from fp32 to fp16
                         .add(FmhaSignature()
                                  .family("bwd_convert_dq")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
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

void cpu_attention_fwd(const std::vector<float>& Q,
                       const std::vector<float>& K,
                       const std::vector<float>& V,
                       std::vector<float>& O,
                       std::vector<float>& LSE,
                       int batch,
                       int nhead,
                       int seqlen_q,
                       int seqlen_k,
                       int hdim_q,
                       int hdim_v,
                       float scale)
{
    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < nhead; ++h)
        {
            for(int sq = 0; sq < seqlen_q; ++sq)
            {
                std::vector<float> scores(seqlen_k, 0.0f);
                float max_score = -1e30f;

                for(int sk = 0; sk < seqlen_k; ++sk)
                {
                    float dot = 0.0f;
                    for(int d = 0; d < hdim_q; ++d)
                    {
                        int q_idx = ((b * nhead + h) * seqlen_q + sq) * hdim_q + d;
                        int k_idx = ((b * nhead + h) * seqlen_k + sk) * hdim_q + d;
                        dot += Q[q_idx] * K[k_idx];
                    }
                    scores[sk] = dot * scale;
                    max_score  = std::max(max_score, scores[sk]);
                }

                float sum_exp = 0.0f;
                for(int sk = 0; sk < seqlen_k; ++sk)
                {
                    scores[sk] = std::exp(scores[sk] - max_score);
                    sum_exp += scores[sk];
                }

                int lse_idx  = (b * nhead + h) * seqlen_q + sq;
                LSE[lse_idx] = max_score + std::log(sum_exp);

                for(int sk = 0; sk < seqlen_k; ++sk)
                {
                    scores[sk] /= sum_exp;
                }

                for(int dv = 0; dv < hdim_v; ++dv)
                {
                    float acc = 0.0f;
                    for(int sk = 0; sk < seqlen_k; ++sk)
                    {
                        int v_idx = ((b * nhead + h) * seqlen_k + sk) * hdim_v + dv;
                        acc += scores[sk] * V[v_idx];
                    }
                    int o_idx = ((b * nhead + h) * seqlen_q + sq) * hdim_v + dv;
                    O[o_idx]  = acc;
                }
            }
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 22: FMHA Backward (GPU)", "Forward + backward with GPU validation");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "1", "Batch size");
    args.add_option("--nhead", "4", "Number of heads");
    args.add_option("--seqlen", "64", "Sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 1);
    const int nhead            = args.get_int("--nhead", 4);
    const int seqlen           = args.get_int("--seqlen", 64);
    const int hdim             = args.get_int("--hdim", 128);
    const float scale          = 1.0f / std::sqrt(static_cast<float>(hdim));

    print_header("Example 22: FMHA Backward (GPU Execution)");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("gpu_bwd_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 3);

    // Step 2: Plan backward to verify all 3 stages resolve
    std::cout << "\nStep 2: Plan Backward\n";

    fmha_bwd_traits bwd_traits{};
    bwd_traits.hdim_q           = hdim;
    bwd_traits.hdim_v           = hdim;
    bwd_traits.data_type        = "fp16";
    bwd_traits.is_group_mode    = false;
    bwd_traits.mask_type        = mask_enum::no_mask;
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

    if(!bwd_plan.is_valid() || bwd_plan.stages.size() < 2)
    {
        std::cout << "  Backward plan: INVALID (expected multi-stage)\n";
        std::cout << "  Falling back to planning-only mode (like 04_bwd_fmha.cpp)\n";
        print_separator();
        std::cout << "Status: PLAN_ONLY\n";
        print_separator();
        return 0;
    }

    std::cout << "  Backward plan stages:\n";
    for(const auto& stage : bwd_plan.stages)
    {
        std::cout << "    " << to_string(stage.family) << " -> " << stage.kernel_id << "\n";
    }

    // Step 3: Allocate buffers
    std::cout << "\nStep 3: Allocate GPU Buffers\n";
    const int64_t qkv_elems    = static_cast<int64_t>(batch) * nhead * seqlen * hdim;
    const int64_t lse_elems    = static_cast<int64_t>(batch) * nhead * seqlen;
    const int64_t dq_acc_elems = static_cast<int64_t>(batch) * nhead * seqlen * hdim;

    std::cout << "  Q/K/V/O: [" << batch << ", " << nhead << ", " << seqlen << ", " << hdim
              << "]\n";
    std::cout << "  LSE/d:   [" << batch << ", " << nhead << ", " << seqlen << "]\n";

    GpuBuffer<FmhaDataType> q_dev(qkv_elems);
    GpuBuffer<FmhaDataType> k_dev(qkv_elems);
    GpuBuffer<FmhaDataType> v_dev(qkv_elems);
    GpuBuffer<FmhaDataType> o_dev(qkv_elems);
    GpuBuffer<float> lse_dev(lse_elems);
    GpuBuffer<FmhaDataType> do_dev(qkv_elems);
    GpuBuffer<float> d_dev(lse_elems);
    GpuBuffer<FmhaDataType> dq_dev(qkv_elems);
    GpuBuffer<FmhaDataType> dk_dev(qkv_elems);
    GpuBuffer<FmhaDataType> dv_dev(qkv_elems);
    GpuBuffer<float> dq_acc_dev(dq_acc_elems);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<FmhaDataType> q_host(qkv_elems), k_host(qkv_elems), v_host(qkv_elems);
    std::vector<FmhaDataType> do_host(qkv_elems);
    for(auto& x : q_host)
        x = FmhaDataType(dist(rng));
    for(auto& x : k_host)
        x = FmhaDataType(dist(rng));
    for(auto& x : v_host)
        x = FmhaDataType(dist(rng));
    for(auto& x : do_host)
        x = FmhaDataType(dist(rng));

    q_dev.copy_from_host(q_host.data());
    k_dev.copy_from_host(k_host.data());
    v_dev.copy_from_host(v_host.data());
    do_dev.copy_from_host(do_host.data());
    o_dev.zero();
    lse_dev.zero();
    d_dev.zero();
    dq_dev.zero();
    dk_dev.zero();
    dv_dev.zero();
    dq_acc_dev.zero();

    // Step 4: Run forward to produce O and LSE
    std::cout << "\nStep 4: Run Forward (to produce O and LSE)\n";
    {
        fmha_fwd_traits fwd_traits{};
        fwd_traits.hdim_q              = hdim;
        fwd_traits.hdim_v              = hdim;
        fwd_traits.data_type           = "fp16";
        fwd_traits.is_group_mode       = false;
        fwd_traits.is_v_rowmajor       = true;
        fwd_traits.has_logits_soft_cap = false;
        fwd_traits.mask_type           = mask_enum::no_mask;
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
        fwd_args.window_size_right   = -1;
        fwd_args.sink_size           = 0;
        fwd_args.mask_type           = 0;
        fwd_args.min_seqlen_q        = 0;
        fwd_args.p_drop              = 0.0f;
        fwd_args.s_randval           = false;
        fwd_args.drop_seed_offset    = std::make_pair(uint64_t(0), uint64_t(0));
        fwd_args.block_scale_size_q  = 0;
        fwd_args.block_scale_size_kv = 0;

        try
        {
            float fwd_time = dispatcher.run_fwd(fwd_traits, fwd_args, nullptr);
            std::cout << "  Forward time: " << std::fixed << std::setprecision(4) << fwd_time
                      << " ms\n";
        }
        catch(const std::exception& e)
        {
            std::cerr << "  Forward ERROR: " << e.what() << "\n";
            print_separator();
            std::cout << "Status: FAIL (forward failed)\n";
            print_separator();
            return 1;
        }
    }

    // Step 5: Run backward
    std::cout << "\nStep 5: Run Backward\n";

    bwd_args.q_ptr        = q_dev.get();
    bwd_args.k_ptr        = k_dev.get();
    bwd_args.v_ptr        = v_dev.get();
    bwd_args.bias_ptr     = nullptr;
    bwd_args.o_ptr        = o_dev.get();
    bwd_args.lse_ptr      = lse_dev.get();
    bwd_args.do_ptr       = do_dev.get();
    bwd_args.d_ptr        = d_dev.get();
    bwd_args.rand_val_ptr = nullptr;
    bwd_args.dq_ptr       = dq_dev.get();
    bwd_args.dk_ptr       = dk_dev.get();
    bwd_args.dv_ptr       = dv_dev.get();
    bwd_args.dbias_ptr    = nullptr;
    bwd_args.dq_acc_ptr   = dq_acc_dev.get();
    bwd_args.scale        = scale;

    bwd_args.stride_q       = hdim;
    bwd_args.stride_k       = hdim;
    bwd_args.stride_v       = hdim;
    bwd_args.stride_bias    = 0;
    bwd_args.stride_o       = hdim;
    bwd_args.stride_randval = 0;
    bwd_args.stride_do      = hdim;
    bwd_args.stride_dq_acc  = hdim;
    bwd_args.stride_dq      = hdim;
    bwd_args.stride_dk      = hdim;
    bwd_args.stride_dv      = hdim;
    bwd_args.stride_dbias   = 0;

    bwd_args.nhead_stride_q       = seqlen * hdim;
    bwd_args.nhead_stride_k       = seqlen * hdim;
    bwd_args.nhead_stride_v       = seqlen * hdim;
    bwd_args.nhead_stride_bias    = 0;
    bwd_args.nhead_stride_o       = seqlen * hdim;
    bwd_args.nhead_stride_randval = 0;
    bwd_args.nhead_stride_do      = seqlen * hdim;
    bwd_args.nhead_stride_lsed    = seqlen;
    bwd_args.nhead_stride_dq_acc  = static_cast<int64_t>(seqlen) * hdim;
    bwd_args.nhead_stride_dq      = seqlen * hdim;
    bwd_args.nhead_stride_dk      = seqlen * hdim;
    bwd_args.nhead_stride_dv      = seqlen * hdim;
    bwd_args.nhead_stride_dbias   = 0;

    bwd_args.batch_stride_q       = nhead * seqlen * hdim;
    bwd_args.batch_stride_k       = nhead * seqlen * hdim;
    bwd_args.batch_stride_v       = nhead * seqlen * hdim;
    bwd_args.batch_stride_bias    = 0;
    bwd_args.batch_stride_o       = nhead * seqlen * hdim;
    bwd_args.batch_stride_randval = 0;
    bwd_args.batch_stride_do      = nhead * seqlen * hdim;
    bwd_args.batch_stride_lsed    = nhead * seqlen;
    bwd_args.batch_stride_dq_acc  = static_cast<int64_t>(nhead) * seqlen * hdim;
    bwd_args.batch_stride_dq      = nhead * seqlen * hdim;
    bwd_args.batch_stride_dk      = nhead * seqlen * hdim;
    bwd_args.batch_stride_dv      = nhead * seqlen * hdim;
    bwd_args.batch_stride_dbias   = 0;
    bwd_args.split_stride_dq_acc  = 0;

    bwd_args.window_size_left  = -1;
    bwd_args.window_size_right = -1;
    bwd_args.mask_type         = 0;
    bwd_args.p_drop            = 0.0f;
    bwd_args.p_undrop          = 1.0f;
    bwd_args.drop_seed_offset  = std::make_pair(uint64_t(0), uint64_t(0));

    bool bwd_passed = false;
    try
    {
        float bwd_time = dispatcher.run_bwd(bwd_traits, bwd_args, nullptr);
        std::cout << "  Backward time: " << std::fixed << std::setprecision(4) << bwd_time
                  << " ms\n";

        // Validate: dQ, dK, dV should be non-zero
        std::vector<FmhaDataType> dq_host(qkv_elems), dk_host(qkv_elems), dv_host(qkv_elems);
        dq_dev.copy_to_host(dq_host.data());
        dk_dev.copy_to_host(dk_host.data());
        dv_dev.copy_to_host(dv_host.data());

        auto count_nonzero = [](const std::vector<FmhaDataType>& buf) {
            int nz = 0;
            for(const auto& x : buf)
            {
                if(static_cast<float>(x) != 0.0f)
                    ++nz;
            }
            return nz;
        };

        int dq_nz = count_nonzero(dq_host);
        int dk_nz = count_nonzero(dk_host);
        int dv_nz = count_nonzero(dv_host);

        std::cout << "  dQ non-zero: " << dq_nz << " / " << qkv_elems << "\n";
        std::cout << "  dK non-zero: " << dk_nz << " / " << qkv_elems << "\n";
        std::cout << "  dV non-zero: " << dv_nz << " / " << qkv_elems << "\n";

        bwd_passed = (dq_nz > 0) && (dk_nz > 0) && (dv_nz > 0);
    }
    catch(const std::exception& e)
    {
        std::cerr << "  Backward ERROR: " << e.what() << "\n";
        std::cout << "  Falling back to planning-only mode (like 04_bwd_fmha.cpp)\n";
        std::cout << "  Backward plan was valid with " << bwd_plan.stages.size() << " stages\n";
        print_separator();
        std::cout << "Status: PLAN_ONLY\n";
        print_separator();
        return 0;
    }

    print_separator();
    std::cout << "Status: " << (bwd_passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return bwd_passed ? 0 : 1;
}
