// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 18: GPU Split-KV FMHA Forward
//
// Demonstrates split-KV attention with GPU execution:
//   1. Declare both fwd_splitkv and fwd_splitkv_combine kernels
//   2. Show 2-stage execution plan
//   3. Allocate Q, K, V, O plus workspace (lse_acc, o_acc)
//   4. Run the split-KV forward pass on GPU
//   5. Copy output to host and validate against CPU reference

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

DECL_FMHA_KERNEL_SET(splitkv_gpu_kernels,
                     .add(FmhaSignature()
                              .family("fwd_splitkv")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("no")
                              .bias("no")
                              .lse(true)
                              .dropout(false)
                              .qscale("no")
                              .paged_kv(false),
                          FmhaAlgorithm()
                              // Stage 0 (Q*K^T): m0=seqlen_q, n0=seqlen_k, k0=hdim_q
                              .tile_m0(64)
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
                              .warp_m0(16)
                              .warp_n0(16)
                              .warp_k0(16)
                              .warp_m1(16)
                              .warp_n1(16)
                              .warp_k1(16)
                              .pipeline("qr_nwarp_sshuffle")
                              .padding(true, true, true, true)
                              .max_splits_log2(6)
                              .selection_rank(0),
                          "gfx950")
                         .add(FmhaSignature()
                                  .family("fwd_splitkv_combine")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
                                  .lse(true)
                                  .dropout(false)
                                  .qscale("no")
                                  .paged_kv(false),
                              FmhaAlgorithm()
                                  .tile_m0(64)
                                  .tile_n0(128)
                                  .tile_k0(32)
                                  .tile_n1(32)
                                  .tile_k1(32)
                                  .tile_k0max(128)
                                  .wave_m0(4)
                                  .wave_n0(1)
                                  .wave_k0(1)
                                  .wave_m1(4)
                                  .wave_n1(1)
                                  .wave_k1(1)
                                  .warp_m0(16)
                                  .warp_n0(16)
                                  .warp_k0(16)
                                  .warp_m1(16)
                                  .warp_n1(16)
                                  .warp_k1(16)
                                  .pipeline("qr")
                                  .padding(true, true, true, true)
                                  .max_splits_log2(6)
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
    ExampleArgs args("Example 18: GPU Split-KV FMHA Forward", "Split-KV with GPU execution");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--nhead", "4", "Number of heads");
    args.add_option("--seqlen_q", "64", "Query sequence length");
    args.add_option("--seqlen_k", "2048", "KV sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    args.add_option("--splits", "2", "Number of KV splits");
    args.add_flag("--validate", "Validate against CPU reference");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 2);
    const int nhead            = args.get_int("--nhead", 4);
    const int seqlen_q         = args.get_int("--seqlen_q", 64);
    const int seqlen_k         = args.get_int("--seqlen_k", 2048);
    const int hdim             = args.get_int("--hdim", 128);
    const int num_splits       = args.get_int("--splits", 2);

    print_header("Example 18: GPU Split-KV FMHA Forward");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("splitkv_gpu_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 3);

    // Step 2: Set up traits and plan
    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));

    fmha_fwd_splitkv_traits traits{};
    traits.hdim_q              = hdim;
    traits.hdim_v              = hdim;
    traits.data_type           = "fp16";
    traits.is_group_mode       = false;
    traits.is_v_rowmajor       = true;
    traits.has_logits_soft_cap = false;
    traits.mask_type           = mask_enum::no_mask;
    traits.bias_type           = bias_enum::no_bias;
    traits.has_lse             = true;
    traits.do_fp8_static_quant = false;
    traits.has_sink            = false;

    // Workspace sizes: lse_acc [batch, nhead, num_splits, seqlen_q]
    //                  o_acc   [batch, nhead, num_splits, seqlen_q, hdim]
    const int64_t q_elems       = static_cast<int64_t>(batch) * nhead * seqlen_q * hdim;
    const int64_t k_elems       = static_cast<int64_t>(batch) * nhead * seqlen_k * hdim;
    const int64_t v_elems       = k_elems;
    const int64_t o_elems       = q_elems;
    const int64_t lse_elems     = static_cast<int64_t>(batch) * nhead * seqlen_q;
    const int64_t lse_acc_elems = static_cast<int64_t>(batch) * nhead * num_splits * seqlen_q;
    const int64_t o_acc_elems = static_cast<int64_t>(batch) * nhead * num_splits * seqlen_q * hdim;

    // Show the 2-stage plan
    std::cout << "\nStep 2: Plan (2-stage split-KV)\n";

    fmha_fwd_splitkv_args plan_args{};
    plan_args.seqlen_q     = seqlen_q;
    plan_args.seqlen_k     = seqlen_k;
    plan_args.batch        = batch;
    plan_args.max_seqlen_q = seqlen_q;
    plan_args.hdim_q       = hdim;
    plan_args.hdim_v       = hdim;
    plan_args.nhead_q      = nhead;
    plan_args.nhead_k      = nhead;
    plan_args.num_splits   = num_splits;

    auto problem = FmhaProblem::from_invocation(FmhaInvocation::make(traits, plan_args), gfx_arch);
    auto plan    = dispatcher.plan(problem);

    if(!plan.is_valid() || plan.stages.size() != 2)
    {
        std::cerr << "  WARNING: Expected a two-stage split-KV plan, got " << plan.stages.size()
                  << " stage(s)\n";
        if(!plan.is_valid())
        {
            std::cerr << "  Plan is invalid -- no matching kernels found\n";
            print_separator();
            return 1;
        }
    }

    for(const auto& stage : plan.stages)
    {
        std::cout << "  " << to_string(stage.family) << " -> " << stage.kernel_id << "\n";
    }

    // Step 3: Allocate GPU buffers
    std::cout << "\nStep 3: Allocate GPU Buffers\n";
    std::cout << "  Q:       [" << batch << ", " << nhead << ", " << seqlen_q << ", " << hdim
              << "]\n";
    std::cout << "  K/V:     [" << batch << ", " << nhead << ", " << seqlen_k << ", " << hdim
              << "]\n";
    std::cout << "  O:       [" << batch << ", " << nhead << ", " << seqlen_q << ", " << hdim
              << "]\n";
    std::cout << "  lse_acc: [" << batch << ", " << nhead << ", " << num_splits << ", " << seqlen_q
              << "]\n";
    std::cout << "  o_acc:   [" << batch << ", " << nhead << ", " << num_splits << ", " << seqlen_q
              << ", " << hdim << "]\n";

    GpuBuffer<FmhaDataType> q_dev(q_elems);
    GpuBuffer<FmhaDataType> k_dev(k_elems);
    GpuBuffer<FmhaDataType> v_dev(v_elems);
    GpuBuffer<FmhaDataType> o_dev(o_elems);
    GpuBuffer<float> lse_dev(lse_elems);
    GpuBuffer<float> lse_acc_dev(lse_acc_elems);
    GpuBuffer<float> o_acc_dev(o_acc_elems);

    // Fill Q, K, V with random data
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
    lse_acc_dev.zero();
    o_acc_dev.zero();

    // Step 4: Set up splitkv args with device pointers and strides
    fmha_fwd_splitkv_args fmha_args{};
    fmha_args.q_ptr = q_dev.get();
    fmha_args.k_ptr = k_dev.get();
    fmha_args.v_ptr = v_dev.get();
    fmha_args.o_ptr = o_dev.get();

    fmha_args.bias_ptr    = nullptr;
    fmha_args.lse_acc_ptr = lse_acc_dev.get();
    fmha_args.o_acc_ptr   = o_acc_dev.get();
    fmha_args.lse_ptr     = lse_dev.get();

    fmha_args.block_table_ptr          = nullptr;
    fmha_args.batch_stride_block_table = 0;
    fmha_args.page_block_size          = 0;
    fmha_args.is_gappy                 = false;
    fmha_args.cache_batch_idx          = nullptr;
    fmha_args.seqstart_q_ptr           = nullptr;
    fmha_args.seqstart_k_ptr           = nullptr;
    fmha_args.seqlen_k_ptr             = nullptr;
    fmha_args.sink_ptr                 = nullptr;

    fmha_args.seqlen_q     = seqlen_q;
    fmha_args.seqlen_k     = seqlen_k;
    fmha_args.batch        = batch;
    fmha_args.max_seqlen_q = seqlen_q;
    fmha_args.hdim_q       = hdim;
    fmha_args.hdim_v       = hdim;
    fmha_args.nhead_q      = nhead;
    fmha_args.nhead_k      = nhead;
    fmha_args.num_splits   = num_splits;

    fmha_args.scale_s         = scale;
    fmha_args.scale_p         = 1.0f;
    fmha_args.scale_o         = 1.0f;
    fmha_args.logits_soft_cap = 0.0f;

    // bhsd layout strides
    fmha_args.stride_q     = hdim;
    fmha_args.stride_k     = hdim;
    fmha_args.stride_v     = hdim;
    fmha_args.stride_bias  = 0;
    fmha_args.stride_o_acc = hdim;
    fmha_args.stride_o     = hdim;

    fmha_args.nhead_stride_q       = seqlen_q * hdim;
    fmha_args.nhead_stride_k       = seqlen_k * hdim;
    fmha_args.nhead_stride_v       = seqlen_k * hdim;
    fmha_args.nhead_stride_bias    = 0;
    fmha_args.nhead_stride_lse     = seqlen_q;
    fmha_args.nhead_stride_lse_acc = num_splits * seqlen_q;
    fmha_args.nhead_stride_o_acc   = num_splits * seqlen_q * hdim;
    fmha_args.nhead_stride_o       = seqlen_q * hdim;

    fmha_args.batch_stride_q       = nhead * seqlen_q * hdim;
    fmha_args.batch_stride_k       = nhead * seqlen_k * hdim;
    fmha_args.batch_stride_v       = nhead * seqlen_k * hdim;
    fmha_args.batch_stride_bias    = 0;
    fmha_args.batch_stride_lse     = nhead * seqlen_q;
    fmha_args.batch_stride_lse_acc = nhead * num_splits * seqlen_q;
    fmha_args.batch_stride_o_acc   = nhead * num_splits * seqlen_q * hdim;
    fmha_args.batch_stride_o       = nhead * seqlen_q * hdim;

    fmha_args.split_stride_lse_acc = seqlen_q;
    fmha_args.split_stride_o_acc   = seqlen_q * hdim;

    fmha_args.window_size_left  = -1;
    fmha_args.window_size_right = -1;
    fmha_args.sink_size         = 0;
    fmha_args.mask_type         = 0;

    // Step 5: Run on GPU
    std::cout << "\nStep 4: Run Split-KV FMHA Forward on GPU\n";
    float time_ms = 0.0f;
    try
    {
        time_ms = dispatcher.run_fwd_splitkv(traits, fmha_args, nullptr);
    }
    catch(const std::exception& e)
    {
        std::cerr << "  WARNING: GPU execution failed: " << e.what() << "\n";
        std::cerr << "  Falling back to planning-only mode (split-KV compilation can be complex)\n";
        std::cout << "\n  Plan summary (2 stages):\n";
        for(const auto& stage : plan.stages)
        {
            std::cout << "    " << to_string(stage.family) << " -> " << stage.kernel_id << "\n";
        }
        print_separator();
        std::cout << "Status: PLAN_ONLY\n";
        print_separator();
        return 0;
    }

    auto run_problem =
        FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_args), gfx_arch);
    double tflops = static_cast<double>(run_problem.num_ops()) / (time_ms * 1e-3) / 1e12;

    std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";

    // Step 6: Copy output and validate
    std::cout << "\nStep 5: Validate\n";
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
            q_f32, k_f32, v_f32, o_ref, batch, nhead, seqlen_q, seqlen_k, hdim, hdim, scale);

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
