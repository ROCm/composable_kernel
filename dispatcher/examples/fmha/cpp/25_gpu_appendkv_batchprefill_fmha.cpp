// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 25: AppendKV + BatchPrefill Planning with GPU Execution
//
// Demonstrates:
//   1. Declare appendkv, batch_prefill, and basic fwd kernels
//   2. Plan appendkv with fmha_fwd_appendkv_traits / fmha_fwd_appendkv_args
//   3. Plan batch_prefill with fmha_batch_prefill_traits / fmha_batch_prefill_args
//   4. Run basic fwd kernel on GPU as sanity check
//   5. Show cache_batch_idx usage pattern for non-contiguous batches
//
// Mirrors 01_basic_fmha.cpp for FMHA.

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

DECL_FMHA_KERNEL_SET(appendkv_batchprefill_kernels,

                     // AppendKV kernel
                     .add(FmhaSignature()
                              .family("fwd_appendkv")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .rope("inter")
                              .paged_kv(true)
                              .kv_cache("vectorized", "sglang", 16),
                          FmhaAlgorithm()
                              // Stage 0 (Q*K^T): m0=seqlen_q, n0=seqlen_k, k0=hdim_q
                              .tile_m0(64)
                              .tile_n0(64)
                              .tile_k0(32)
                              // Stage 1 (Attn*V): n1=hdim_v, k1=seqlen_k, k0max=alignment
                              .tile_n1(128)
                              .tile_k1(32)
                              .tile_k0max(128)
                              .pipeline("appendkv")
                              .padding(true, true, true, true)
                              .selection_rank(0),
                          "gfx950")

                         // BatchPrefill kernel (group mode, paged KV, page_size=64)
                         .add(FmhaSignature()
                                  .family("batch_prefill")
                                  .dtype("fp16")
                                  .mode("group")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
                                  .lse(true)
                                  .dropout(false)
                                  .qscale("no")
                                  .paged_kv(true)
                                  .kv_cache("vectorized", "sglang", 64),
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
                                  .selection_rank(0),
                              "gfx950")

                         // Basic fwd kernel for GPU execution sanity check
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
    ExampleArgs args("Example 25: AppendKV + BatchPrefill + GPU",
                     "FMHA AppendKV/BatchPrefill planning with GPU sanity check");
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

    print_header("Example 25: AppendKV + BatchPrefill + GPU Execution");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("appendkv_batchprefill");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 3);

    // =========================================================================
    // Step 2: Plan AppendKV
    //   traits: fmha_fwd_appendkv_traits (hdim_q, hdim_v, data_type,
    //           is_v_rowmajor, rope_type)
    //   args:   fmha_fwd_appendkv_args (q_ptr, k_ptr, knew_ptr, v_ptr,
    //           vnew_ptr, seqlen_q, seqlen_knew, ...)
    // =========================================================================
    std::cout << "\nStep 2: Plan AppendKV\n";

    fmha_fwd_appendkv_traits append_traits{};
    append_traits.hdim_q        = hdim;
    append_traits.hdim_v        = hdim;
    append_traits.data_type     = "fp16";
    append_traits.is_v_rowmajor = true;
    append_traits.rope_type     = rope_enum::interleaved;

    fmha_fwd_appendkv_args append_args{};
    append_args.q_ptr           = reinterpret_cast<void*>(0x1);
    append_args.k_ptr           = reinterpret_cast<void*>(0x1);
    append_args.knew_ptr        = reinterpret_cast<void*>(0x1);
    append_args.v_ptr           = reinterpret_cast<void*>(0x1);
    append_args.vnew_ptr        = reinterpret_cast<void*>(0x1);
    append_args.seqlen_q        = 1;
    append_args.seqlen_knew     = 1;
    append_args.batch           = batch;
    append_args.hdim_q          = hdim;
    append_args.hdim_v          = hdim;
    append_args.nhead_q         = nhead;
    append_args.nhead_k         = nhead;
    append_args.rotary_dim      = hdim;
    append_args.rotary_cos_ptr  = reinterpret_cast<void*>(0x1);
    append_args.rotary_sin_ptr  = reinterpret_cast<void*>(0x1);
    append_args.block_table_ptr = reinterpret_cast<void*>(0x1);
    append_args.page_block_size = 16;

    // cache_batch_idx: maps request index -> cache slot for non-contiguous batches.
    // When serving multiple requests that don't occupy contiguous cache slots,
    // this indirection array tells the kernel which cache row each request maps to.
    append_args.cache_batch_idx_ptr = reinterpret_cast<void*>(0x1);

    auto append_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(append_traits, append_args), gfx_arch));

    std::cout << "  AppendKV plan valid: " << (append_plan.is_valid() ? "yes" : "no") << "\n";
    if(append_plan.is_valid())
    {
        for(const auto& stage : append_plan.stages)
            std::cout << "  " << to_string(stage.family) << " -> " << stage.kernel_id << "\n";
    }

    // =========================================================================
    // Step 3: Plan BatchPrefill
    //   traits: fmha_batch_prefill_traits (extends fmha_fwd_traits with
    //           kv_memory_layout, kv_lookup_table, page_size)
    //   args:   fmha_batch_prefill_args (kv_indptr, kv_page_indices,
    //           kv_last_page_lens, seqstart_q_ptr, ...)
    // =========================================================================
    std::cout << "\nStep 3: Plan BatchPrefill\n";

    fmha_batch_prefill_traits prefill_traits{};
    prefill_traits.hdim_q        = hdim;
    prefill_traits.hdim_v        = hdim;
    prefill_traits.data_type     = "fp16";
    prefill_traits.is_group_mode = true;
    prefill_traits.is_v_rowmajor = true;
    prefill_traits.mask_type     = mask_enum::no_mask;
    prefill_traits.bias_type     = bias_enum::no_bias;
    prefill_traits.has_lse       = true;
    prefill_traits.kv_memory_layout =
        ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    prefill_traits.kv_lookup_table =
        ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    prefill_traits.page_size = 64;

    fmha_batch_prefill_args prefill_args{};
    prefill_args.batch           = batch;
    prefill_args.seqlen_q        = seqlen;
    prefill_args.seqlen_k        = 1024;
    prefill_args.max_seqlen_q    = seqlen;
    prefill_args.hdim_q          = hdim;
    prefill_args.hdim_v          = hdim;
    prefill_args.nhead_q         = nhead;
    prefill_args.nhead_k         = nhead;
    prefill_args.num_total_pages = 128;
    prefill_args.page_block_size = 64;
    prefill_args.kv_memory_layout =
        ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    prefill_args.kv_lookup_table =
        ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    prefill_args.kv_indptr         = reinterpret_cast<void*>(0x1);
    prefill_args.kv_page_indices   = reinterpret_cast<void*>(0x1);
    prefill_args.kv_last_page_lens = reinterpret_cast<void*>(0x1);
    prefill_args.seqstart_q_ptr    = reinterpret_cast<void*>(0x1);

    auto prefill_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(prefill_traits, prefill_args), gfx_arch));

    std::cout << "  BatchPrefill plan valid: " << (prefill_plan.is_valid() ? "yes" : "no") << "\n";
    if(prefill_plan.is_valid())
    {
        for(const auto& stage : prefill_plan.stages)
            std::cout << "  " << to_string(stage.family) << " -> " << stage.kernel_id << "\n";
    }

    // =========================================================================
    // Step 4: GPU Execution with basic fwd kernel (sanity check)
    // =========================================================================
    std::cout << "\nStep 4: Allocate GPU Buffers\n";

    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));

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

    // Step 5: Run on GPU
    std::cout << "\nStep 5: Run FMHA Forward on GPU\n";
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

    // Step 6: Validate
    std::cout << "\nStep 6: Validate\n";
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
