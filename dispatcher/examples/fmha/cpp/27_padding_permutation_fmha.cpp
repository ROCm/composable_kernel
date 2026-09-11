// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 27: Padding, Group Mode, V Col-Major, Permutation Patterns
//
// Demonstrates:
//   1. Batch padding with cu_seqlen arrays for per-batch variable lengths
//   2. Group mode with seqstart_q / seqstart_k buffers
//   3. V col-major layout declaration: .vlayout("c")
//   4. Permutation patterns: bhsd (iperm=1) vs bshd (iperm=0) strides
//   5. GPU execution with basic kernel + batch padding
//
// Mirrors 01_basic_fmha.cpp for FMHA.

#include <hip/hip_runtime.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;

DECL_FMHA_KERNEL_SET(padding_permutation_kernels,

                     // Basic fwd kernel (batch mode, for GPU execution)
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("no")
                              .bias("no")
                              .lse(false)
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

                         // Group mode kernel (variable-length sequences)
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("group")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
                                  .lse(false)
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

                         // V col-major layout declaration
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("c")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
                                  .lse(false)
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
                              "gfx950"));

namespace {

using FmhaDataType = ck_tile::fp16_t;

void cpu_attention_fwd(const std::vector<float>& Q,
                       const std::vector<float>& K,
                       const std::vector<float>& V,
                       std::vector<float>& O,
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
    ExampleArgs args("Example 27: Padding & Permutation FMHA",
                     "FMHA padding, group mode, V col-major, and permutation patterns");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--nhead", "4", "Number of heads");
    args.add_option("--seqlen", "64", "Sequence length (Q and K)");
    args.add_option("--hdim", "128", "Head dimension");
    args.add_flag("--validate", "Validate against CPU reference");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 2);
    const int nhead            = args.get_int("--nhead", 4);
    const int seqlen           = args.get_int("--seqlen", 64);
    const int hdim             = args.get_int("--hdim", 128);

    print_header("Example 27: Padding, Group Mode, V Col-Major, Permutation");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("padding_permutation");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 3);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));

    // =========================================================================
    // Step 2: Batch Padding Pattern
    //   Allocate cu_seqlen_q / cu_seqlen_k buffers with cumulative sums.
    //   In CK's dispatcher, this maps to seqstart_q_ptr / seqstart_k_ptr
    //   and requires group mode to enable per-batch variable sequence lengths.
    // =========================================================================
    std::cout << "\nStep 2: Batch Padding Pattern (cu_seqlen)\n";
    {
        // Per-batch sequence lengths: batch 0 has seqlen=32, batch 1 has seqlen=48
        const std::vector<int32_t> seqlens_q = {32, 48};
        const std::vector<int32_t> seqlens_k = {32, 48};
        const int num_batches                = static_cast<int>(seqlens_q.size());

        // Build cumulative sum arrays: [0, 32, 80]
        std::vector<int32_t> cu_seqlen_q(num_batches + 1, 0);
        std::vector<int32_t> cu_seqlen_k(num_batches + 1, 0);
        for(int i = 0; i < num_batches; ++i)
        {
            cu_seqlen_q[i + 1] = cu_seqlen_q[i] + seqlens_q[i];
            cu_seqlen_k[i + 1] = cu_seqlen_k[i] + seqlens_k[i];
        }

        const int total_q = cu_seqlen_q.back();
        const int total_k = cu_seqlen_k.back();
        const int max_sq  = *std::max_element(seqlens_q.begin(), seqlens_q.end());

        std::cout << "  Batch seqlens_q: [";
        for(int i = 0; i < num_batches; ++i)
            std::cout << (i ? ", " : "") << seqlens_q[i];
        std::cout << "]\n";
        std::cout << "  cu_seqlen_q:     [";
        for(size_t i = 0; i < cu_seqlen_q.size(); ++i)
            std::cout << (i ? ", " : "") << cu_seqlen_q[i];
        std::cout << "]\n";

        GpuBuffer<int32_t> cu_sq_dev(num_batches + 1);
        GpuBuffer<int32_t> cu_sk_dev(num_batches + 1);
        cu_sq_dev.copy_from_host(cu_seqlen_q.data());
        cu_sk_dev.copy_from_host(cu_seqlen_k.data());

        // Group mode traits for variable-length sequences
        fmha_fwd_traits pad_traits{};
        pad_traits.hdim_q              = hdim;
        pad_traits.hdim_v              = hdim;
        pad_traits.data_type           = "fp16";
        pad_traits.is_group_mode       = true;
        pad_traits.is_v_rowmajor       = true;
        pad_traits.has_logits_soft_cap = false;
        pad_traits.mask_type           = mask_enum::no_mask;
        pad_traits.bias_type           = bias_enum::no_bias;
        pad_traits.has_lse             = false;
        pad_traits.has_dropout         = false;
        pad_traits.qscale_type         = quant_scale_enum::no_scale;

        fmha_fwd_args pad_args{};
        pad_args.seqlen_q     = total_q;
        pad_args.seqlen_k     = total_k;
        pad_args.batch        = num_batches;
        pad_args.max_seqlen_q = max_sq;
        pad_args.hdim_q       = hdim;
        pad_args.hdim_v       = hdim;
        pad_args.nhead_q      = nhead;
        pad_args.nhead_k      = nhead;
        pad_args.scale_s      = scale;

        // cu_seqlen_q_ptr / cu_seqlen_k_ptr (seqstart_q / seqstart_k in CK)
        pad_args.seqstart_q_ptr = cu_sq_dev.get();
        pad_args.seqstart_k_ptr = cu_sk_dev.get();

        auto pad_plan = dispatcher.plan(
            FmhaProblem::from_invocation(FmhaInvocation::make(pad_traits, pad_args), gfx_arch));
        std::cout << "  Batch padding plan valid: " << (pad_plan.is_valid() ? "yes" : "no") << "\n";
    }

    // =========================================================================
    // Step 3: Group Mode Pattern
    //   Group mode uses seqstart_q / seqstart_k arrays to define variable
    //   sequence boundaries. Each batch element can have a different length.
    //   traits.is_group_mode = true
    // =========================================================================
    std::cout << "\nStep 3: Group Mode Pattern (seqstart)\n";
    {
        fmha_fwd_traits group_traits{};
        group_traits.hdim_q              = hdim;
        group_traits.hdim_v              = hdim;
        group_traits.data_type           = "fp16";
        group_traits.is_group_mode       = true;
        group_traits.is_v_rowmajor       = true;
        group_traits.has_logits_soft_cap = false;
        group_traits.mask_type           = mask_enum::no_mask;
        group_traits.bias_type           = bias_enum::no_bias;
        group_traits.has_lse             = false;
        group_traits.has_dropout         = false;
        group_traits.qscale_type         = quant_scale_enum::no_scale;

        const std::vector<int32_t> seqstart_q = {0, 64, 192};
        const std::vector<int32_t> seqstart_k = {0, 128, 256};
        const int num_batches                 = static_cast<int>(seqstart_q.size()) - 1;
        const int total_q                     = seqstart_q.back();
        const int max_sq                      = 128;

        GpuBuffer<int32_t> ss_q_dev(seqstart_q.size());
        GpuBuffer<int32_t> ss_k_dev(seqstart_k.size());
        ss_q_dev.copy_from_host(seqstart_q.data());
        ss_k_dev.copy_from_host(seqstart_k.data());

        fmha_fwd_args group_args{};
        group_args.seqlen_q       = total_q;
        group_args.seqlen_k       = seqstart_k.back();
        group_args.batch          = num_batches;
        group_args.max_seqlen_q   = max_sq;
        group_args.hdim_q         = hdim;
        group_args.hdim_v         = hdim;
        group_args.nhead_q        = nhead;
        group_args.nhead_k        = nhead;
        group_args.scale_s        = scale;
        group_args.seqstart_q_ptr = ss_q_dev.get();
        group_args.seqstart_k_ptr = ss_k_dev.get();

        std::cout << "  seqstart_q: [0, 64, 192] -> batches of length 64 and 128\n";
        std::cout << "  seqstart_k: [0, 128, 256] -> KV of length 128 and 128\n";

        auto group_plan = dispatcher.plan(
            FmhaProblem::from_invocation(FmhaInvocation::make(group_traits, group_args), gfx_arch));
        std::cout << "  Group mode plan valid: " << (group_plan.is_valid() ? "yes" : "no") << "\n";
    }

    // =========================================================================
    // Step 4: V Col-Major Declaration
    //   .vlayout("c") declares V in column-major layout (seqlen_k x hdim_v
    //   stored column-first). This affects how the kernel reads V.
    // =========================================================================
    std::cout << "\nStep 4: V Col-Major Layout\n";
    {
        fmha_fwd_traits vcol_traits{};
        vcol_traits.hdim_q              = hdim;
        vcol_traits.hdim_v              = hdim;
        vcol_traits.data_type           = "fp16";
        vcol_traits.is_group_mode       = false;
        vcol_traits.is_v_rowmajor       = false;
        vcol_traits.has_logits_soft_cap = false;
        vcol_traits.mask_type           = mask_enum::no_mask;
        vcol_traits.bias_type           = bias_enum::no_bias;
        vcol_traits.has_lse             = false;
        vcol_traits.has_dropout         = false;
        vcol_traits.qscale_type         = quant_scale_enum::no_scale;

        fmha_fwd_args vcol_args{};
        vcol_args.batch        = batch;
        vcol_args.seqlen_q     = seqlen;
        vcol_args.seqlen_k     = seqlen;
        vcol_args.max_seqlen_q = seqlen;
        vcol_args.hdim_q       = hdim;
        vcol_args.hdim_v       = hdim;
        vcol_args.nhead_q      = nhead;
        vcol_args.nhead_k      = nhead;
        vcol_args.scale_s      = scale;

        std::cout << "  V row-major (.vlayout(\"r\")): stride_v = hdim, "
                     "contiguous along head dimension\n";
        std::cout << "  V col-major (.vlayout(\"c\")): stride_v = seqlen_k, "
                     "contiguous along sequence dimension\n";
        std::cout << "  traits.is_v_rowmajor = false\n";

        auto vcol_plan = dispatcher.plan(
            FmhaProblem::from_invocation(FmhaInvocation::make(vcol_traits, vcol_args), gfx_arch));
        std::cout << "  V col-major plan valid: " << (vcol_plan.is_valid() ? "yes" : "no") << "\n";
    }

    // =========================================================================
    // Step 5: Permutation Patterns (bhsd vs bshd)
    //   bhsd layout (iperm=1): stride_q = hdim, nhead_stride_q = seqlen*hdim
    //     memory: [batch, head, seq, dim]
    //   bshd layout (iperm=0): stride_q = nhead*hdim, nhead_stride_q = hdim
    //     memory: [batch, seq, head, dim]
    // =========================================================================
    std::cout << "\nStep 5: Permutation Patterns\n";
    {
        std::cout << "  bhsd layout (iperm=1):\n";
        std::cout << "    stride_q       = hdim                 = " << hdim << "\n";
        std::cout << "    nhead_stride_q = seqlen * hdim        = " << seqlen * hdim << "\n";
        std::cout << "    batch_stride_q = nhead * seqlen * hdim = " << nhead * seqlen * hdim
                  << "\n";
        std::cout << "    memory order: [batch, head, seq, dim]\n";

        std::cout << "\n  bshd layout (iperm=0):\n";
        std::cout << "    stride_q       = nhead * hdim         = " << nhead * hdim << "\n";
        std::cout << "    nhead_stride_q = hdim                 = " << hdim << "\n";
        std::cout << "    batch_stride_q = seqlen * nhead * hdim = " << seqlen * nhead * hdim
                  << "\n";
        std::cout << "    memory order: [batch, seq, head, dim]\n";
    }

    // =========================================================================
    // Step 6: GPU Execution with basic kernel + batch padding
    //   Run the batch-mode kernel with a non-tile-aligned seqlen to exercise
    //   the .padding(true, true, true, true) capability.
    // =========================================================================
    std::cout << "\nStep 6: GPU Execution (batch mode, seqlen=" << seqlen << ")\n";

    fmha_fwd_traits fwd_traits{};
    fwd_traits.hdim_q              = hdim;
    fwd_traits.hdim_v              = hdim;
    fwd_traits.data_type           = "fp16";
    fwd_traits.is_group_mode       = false;
    fwd_traits.is_v_rowmajor       = true;
    fwd_traits.has_logits_soft_cap = false;
    fwd_traits.mask_type           = mask_enum::no_mask;
    fwd_traits.bias_type           = bias_enum::no_bias;
    fwd_traits.has_lse             = false;
    fwd_traits.has_dropout         = false;
    fwd_traits.qscale_type         = quant_scale_enum::no_scale;

    const int64_t q_elems = static_cast<int64_t>(batch) * nhead * seqlen * hdim;
    const int64_t k_elems = q_elems;
    const int64_t v_elems = q_elems;
    const int64_t o_elems = q_elems;

    std::cout << "  Q/K/V/O: [" << batch << ", " << nhead << ", " << seqlen << ", " << hdim
              << "]\n";

    GpuBuffer<FmhaDataType> q_dev(q_elems);
    GpuBuffer<FmhaDataType> k_dev(k_elems);
    GpuBuffer<FmhaDataType> v_dev(v_elems);
    GpuBuffer<FmhaDataType> o_dev(o_elems);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<FmhaDataType> q_host(q_elems);
    std::vector<FmhaDataType> k_host(k_elems);
    std::vector<FmhaDataType> v_host(v_elems);
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

    fmha_fwd_args fwd_args{};
    fwd_args.q_ptr = q_dev.get();
    fwd_args.k_ptr = k_dev.get();
    fwd_args.v_ptr = v_dev.get();
    fwd_args.o_ptr = o_dev.get();

    fwd_args.bias_ptr                   = nullptr;
    fwd_args.q_descale_ptr              = nullptr;
    fwd_args.k_descale_ptr              = nullptr;
    fwd_args.v_descale_ptr              = nullptr;
    fwd_args.rand_val_ptr               = nullptr;
    fwd_args.lse_ptr                    = nullptr;
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

    // bhsd layout strides (iperm=1)
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
    fwd_args.nhead_stride_lse       = 0;
    fwd_args.nhead_stride_o         = seqlen * hdim;
    fwd_args.nhead_stride_q_descale = 0;
    fwd_args.nhead_stride_k_descale = 0;
    fwd_args.nhead_stride_v_descale = 0;

    fwd_args.batch_stride_q         = nhead * seqlen * hdim;
    fwd_args.batch_stride_k         = nhead * seqlen * hdim;
    fwd_args.batch_stride_v         = nhead * seqlen * hdim;
    fwd_args.batch_stride_bias      = 0;
    fwd_args.batch_stride_randval   = 0;
    fwd_args.batch_stride_lse       = 0;
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

    float time_ms = 0.0f;
    try
    {
        time_ms = dispatcher.run_fwd(fwd_traits, fwd_args, nullptr);
    }
    catch(const std::exception& e)
    {
        std::cerr << "  ERROR: " << e.what() << "\n";
        return 1;
    }

    auto problem =
        FmhaProblem::from_invocation(FmhaInvocation::make(fwd_traits, fwd_args), gfx_arch);
    double tflops = static_cast<double>(problem.num_ops()) / (time_ms * 1e-3) / 1e12;

    std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";

    // Step 7: Validate
    std::cout << "\nStep 7: Validate\n";
    std::vector<FmhaDataType> o_host(o_elems);
    o_dev.copy_to_host(o_host.data());

    int nonzero = 0;
    for(int64_t i = 0; i < o_elems; ++i)
    {
        if(static_cast<float>(o_host[i]) != 0.0f)
            ++nonzero;
    }
    std::cout << "  Non-zero outputs: " << nonzero << " / " << o_elems << "\n";

    bool passed = (nonzero > 0);

    if(args.has("--validate"))
    {
        std::vector<float> q_f32(q_elems), k_f32(k_elems), v_f32(v_elems), o_ref(o_elems, 0.0f);
        for(int64_t i = 0; i < q_elems; ++i)
            q_f32[i] = static_cast<float>(q_host[i]);
        for(int64_t i = 0; i < k_elems; ++i)
            k_f32[i] = static_cast<float>(k_host[i]);
        for(int64_t i = 0; i < v_elems; ++i)
            v_f32[i] = static_cast<float>(v_host[i]);

        cpu_attention_fwd(
            q_f32, k_f32, v_f32, o_ref, batch, nhead, seqlen, seqlen, hdim, hdim, scale);

        double max_abs_err = 0.0;
        double max_rel_err = 0.0;
        int errors         = 0;
        const double rtol  = 1e-2;
        const double atol  = 1e-2;

        for(int64_t i = 0; i < o_elems; ++i)
        {
            float gpu_val  = static_cast<float>(o_host[i]);
            float ref_val  = o_ref[i];
            double abs_err = std::abs(gpu_val - ref_val);
            double rel_err = abs_err / (std::abs(ref_val) + 1e-6);
            max_abs_err    = std::max(max_abs_err, abs_err);
            max_rel_err    = std::max(max_rel_err, rel_err);
            if(abs_err > atol + rtol * std::abs(ref_val))
                ++errors;
        }

        std::cout << "  Max abs error: " << std::scientific << max_abs_err << "\n";
        std::cout << "  Max rel error: " << max_rel_err << "\n";
        std::cout << "  Errors: " << errors << " / " << o_elems << "\n";
        passed = (errors == 0);
    }

    print_separator();
    std::cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return passed ? 0 : 1;
}
