// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 21: GPU Features FMHA
//
// Tests multiple FMHA features with real GPU execution:
//   1. Dropout (with LSE, rand_val buffer)
//   2. GQA (nhead_q=16, nhead_k=4, same kernel)
//   3. LSE output (verify log-sum-exp values)
//
// Mirrors 01_basic_fmha.cpp for each feature variant.

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

DECL_FMHA_KERNEL_SET(gpu_features_fmha_kernels,
                     // Basic fp16 kernel (used for GQA -- GQA is a runtime concern, same kernel)
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

                         // Dropout kernel (requires LSE)
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
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

                         // LSE-only kernel
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
                       int nhead_q,
                       int nhead_k,
                       int seqlen_q,
                       int seqlen_k,
                       int hdim_q,
                       int hdim_v,
                       float scale,
                       std::vector<float>* lse_out = nullptr)
{
    const int nhead_ratio = nhead_q / nhead_k;

    for(int b = 0; b < batch; ++b)
    {
        for(int hq = 0; hq < nhead_q; ++hq)
        {
            const int hk = hq / nhead_ratio;

            for(int sq = 0; sq < seqlen_q; ++sq)
            {
                std::vector<float> scores(seqlen_k, 0.0f);
                float max_score = -1e30f;

                for(int sk = 0; sk < seqlen_k; ++sk)
                {
                    float dot = 0.0f;
                    for(int d = 0; d < hdim_q; ++d)
                    {
                        int q_idx = ((b * nhead_q + hq) * seqlen_q + sq) * hdim_q + d;
                        int k_idx = ((b * nhead_k + hk) * seqlen_k + sk) * hdim_q + d;
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

                if(lse_out)
                {
                    int lse_idx         = (b * nhead_q + hq) * seqlen_q + sq;
                    (*lse_out)[lse_idx] = max_score + std::log(sum_exp);
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
                        int v_idx = ((b * nhead_k + hk) * seqlen_k + sk) * hdim_v + dv;
                        acc += scores[sk] * V[v_idx];
                    }
                    int o_idx = ((b * nhead_q + hq) * seqlen_q + sq) * hdim_v + dv;
                    O[o_idx]  = acc;
                }
            }
        }
    }
}

struct FeatureResult
{
    std::string name;
    bool passed;
    float time_ms;
};

fmha_fwd_args make_base_args(void* q,
                             void* k,
                             void* v,
                             void* o,
                             int batch,
                             int nhead_q,
                             int nhead_k,
                             int seqlen,
                             int hdim,
                             float scale)
{
    fmha_fwd_args a{};
    a.q_ptr = q;
    a.k_ptr = k;
    a.v_ptr = v;
    a.o_ptr = o;

    a.bias_ptr                   = nullptr;
    a.q_descale_ptr              = nullptr;
    a.k_descale_ptr              = nullptr;
    a.v_descale_ptr              = nullptr;
    a.rand_val_ptr               = nullptr;
    a.lse_ptr                    = nullptr;
    a.sink_ptr                   = nullptr;
    a.block_scale_seqstart_q_ptr = nullptr;
    a.block_scale_seqstart_k_ptr = nullptr;

    a.seqlen_q        = seqlen;
    a.seqlen_k        = seqlen;
    a.batch           = batch;
    a.max_seqlen_q    = seqlen;
    a.hdim_q          = hdim;
    a.hdim_v          = hdim;
    a.nhead_q         = nhead_q;
    a.nhead_k         = nhead_k;
    a.scale_s         = scale;
    a.logits_soft_cap = 0.0f;

    a.stride_q       = hdim;
    a.stride_k       = hdim;
    a.stride_v       = hdim;
    a.stride_bias    = 0;
    a.stride_randval = 0;
    a.stride_o       = hdim;

    a.nhead_stride_q         = seqlen * hdim;
    a.nhead_stride_k         = seqlen * hdim;
    a.nhead_stride_v         = seqlen * hdim;
    a.nhead_stride_bias      = 0;
    a.nhead_stride_randval   = 0;
    a.nhead_stride_lse       = 0;
    a.nhead_stride_o         = seqlen * hdim;
    a.nhead_stride_q_descale = 0;
    a.nhead_stride_k_descale = 0;
    a.nhead_stride_v_descale = 0;

    a.batch_stride_q         = nhead_q * seqlen * hdim;
    a.batch_stride_k         = nhead_k * seqlen * hdim;
    a.batch_stride_v         = nhead_k * seqlen * hdim;
    a.batch_stride_bias      = 0;
    a.batch_stride_randval   = 0;
    a.batch_stride_lse       = 0;
    a.batch_stride_o         = nhead_q * seqlen * hdim;
    a.batch_stride_q_descale = 0;
    a.batch_stride_k_descale = 0;
    a.batch_stride_v_descale = 0;

    a.window_size_left    = -1;
    a.window_size_right   = -1;
    a.sink_size           = 0;
    a.mask_type           = 0;
    a.min_seqlen_q        = 0;
    a.p_drop              = 0.0f;
    a.s_randval           = false;
    a.drop_seed_offset    = std::make_pair(uint64_t(0), uint64_t(0));
    a.block_scale_size_q  = 0;
    a.block_scale_size_kv = 0;

    return a;
}

} // namespace

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 21: GPU Features FMHA", "Dropout, GQA, LSE with real GPU data");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--seqlen", "64", "Sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 2);
    const int seqlen           = args.get_int("--seqlen", 64);
    const int hdim             = args.get_int("--hdim", 128);
    const float scale          = 1.0f / std::sqrt(static_cast<float>(hdim));

    print_header("Example 21: GPU Features FMHA");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("gpu_features_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 3);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<FeatureResult> results;

    // -----------------------------------------------------------------------
    // Feature A: GQA (nhead_q=16, nhead_k=4, same basic kernel)
    // -----------------------------------------------------------------------
    {
        std::cout << "\nStep 2a: GQA (nhead_q=16, nhead_k=4)\n";
        const int nhead_q = 16;
        const int nhead_k = 4;

        const int64_t q_elems = static_cast<int64_t>(batch) * nhead_q * seqlen * hdim;
        const int64_t k_elems = static_cast<int64_t>(batch) * nhead_k * seqlen * hdim;
        const int64_t o_elems = q_elems;

        GpuBuffer<FmhaDataType> q_dev(q_elems);
        GpuBuffer<FmhaDataType> k_dev(k_elems);
        GpuBuffer<FmhaDataType> v_dev(k_elems);
        GpuBuffer<FmhaDataType> o_dev(o_elems);

        std::vector<FmhaDataType> q_host(q_elems), k_host(k_elems), v_host(k_elems);
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

        fmha_fwd_traits traits{};
        traits.hdim_q              = hdim;
        traits.hdim_v              = hdim;
        traits.data_type           = "fp16";
        traits.is_group_mode       = false;
        traits.is_v_rowmajor       = true;
        traits.has_logits_soft_cap = false;
        traits.mask_type           = mask_enum::no_mask;
        traits.bias_type           = bias_enum::no_bias;
        traits.has_lse             = false;
        traits.has_dropout         = false;
        traits.qscale_type         = quant_scale_enum::no_scale;

        auto fmha_args = make_base_args(q_dev.get(),
                                        k_dev.get(),
                                        v_dev.get(),
                                        o_dev.get(),
                                        batch,
                                        nhead_q,
                                        nhead_k,
                                        seqlen,
                                        hdim,
                                        scale);

        bool passed   = false;
        float time_ms = 0.0f;
        try
        {
            time_ms = dispatcher.run_fwd(traits, fmha_args, nullptr);
            std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";

            // Validate against CPU reference with GQA head repetition
            std::vector<float> q_f32(q_elems), k_f32(k_elems), v_f32(k_elems);
            for(int64_t i = 0; i < q_elems; ++i)
                q_f32[i] = static_cast<float>(q_host[i]);
            for(int64_t i = 0; i < k_elems; ++i)
                k_f32[i] = static_cast<float>(k_host[i]);
            for(int64_t i = 0; i < k_elems; ++i)
                v_f32[i] = static_cast<float>(v_host[i]);

            std::vector<float> o_ref(o_elems, 0.0f);
            cpu_attention_fwd(q_f32,
                              k_f32,
                              v_f32,
                              o_ref,
                              batch,
                              nhead_q,
                              nhead_k,
                              seqlen,
                              seqlen,
                              hdim,
                              hdim,
                              scale);

            std::vector<FmhaDataType> o_host(o_elems);
            o_dev.copy_to_host(o_host.data());

            double max_abs_err = 0.0;
            int errors         = 0;
            const double rtol  = 1e-2;
            const double atol  = 1e-2;
            for(int64_t i = 0; i < o_elems; ++i)
            {
                float gpu_val  = static_cast<float>(o_host[i]);
                float ref_val  = o_ref[i];
                double abs_err = std::abs(gpu_val - ref_val);
                max_abs_err    = std::max(max_abs_err, abs_err);
                if(abs_err > atol + rtol * std::abs(ref_val))
                    ++errors;
            }
            std::cout << "  Max abs error: " << std::scientific << max_abs_err << "\n";
            std::cout << "  Errors: " << errors << " / " << o_elems << "\n";
            passed = (errors == 0);
        }
        catch(const std::exception& e)
        {
            std::cerr << "  ERROR: " << e.what() << "\n";
        }
        results.push_back({"GQA (16q/4k)", passed, time_ms});
    }

    // -----------------------------------------------------------------------
    // Feature B: LSE output
    // -----------------------------------------------------------------------
    {
        std::cout << "\nStep 2b: LSE Output\n";
        const int nhead = 4;

        const int64_t qkv_elems = static_cast<int64_t>(batch) * nhead * seqlen * hdim;
        const int64_t lse_elems = static_cast<int64_t>(batch) * nhead * seqlen;

        GpuBuffer<FmhaDataType> q_dev(qkv_elems);
        GpuBuffer<FmhaDataType> k_dev(qkv_elems);
        GpuBuffer<FmhaDataType> v_dev(qkv_elems);
        GpuBuffer<FmhaDataType> o_dev(qkv_elems);
        GpuBuffer<float> lse_dev(lse_elems);

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

        fmha_fwd_traits traits{};
        traits.hdim_q              = hdim;
        traits.hdim_v              = hdim;
        traits.data_type           = "fp16";
        traits.is_group_mode       = false;
        traits.is_v_rowmajor       = true;
        traits.has_logits_soft_cap = false;
        traits.mask_type           = mask_enum::no_mask;
        traits.bias_type           = bias_enum::no_bias;
        traits.has_lse             = true;
        traits.has_dropout         = false;
        traits.qscale_type         = quant_scale_enum::no_scale;

        auto fmha_args             = make_base_args(q_dev.get(),
                                        k_dev.get(),
                                        v_dev.get(),
                                        o_dev.get(),
                                        batch,
                                        nhead,
                                        nhead,
                                        seqlen,
                                        hdim,
                                        scale);
        fmha_args.lse_ptr          = lse_dev.get();
        fmha_args.nhead_stride_lse = seqlen;
        fmha_args.batch_stride_lse = nhead * seqlen;

        bool passed   = false;
        float time_ms = 0.0f;
        try
        {
            time_ms = dispatcher.run_fwd(traits, fmha_args, nullptr);
            std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";

            // Compute CPU reference LSE
            std::vector<float> q_f32(qkv_elems), k_f32(qkv_elems), v_f32(qkv_elems);
            for(int64_t i = 0; i < qkv_elems; ++i)
                q_f32[i] = static_cast<float>(q_host[i]);
            for(int64_t i = 0; i < qkv_elems; ++i)
                k_f32[i] = static_cast<float>(k_host[i]);
            for(int64_t i = 0; i < qkv_elems; ++i)
                v_f32[i] = static_cast<float>(v_host[i]);

            std::vector<float> o_ref(qkv_elems, 0.0f);
            std::vector<float> lse_ref(lse_elems, 0.0f);
            cpu_attention_fwd(q_f32,
                              k_f32,
                              v_f32,
                              o_ref,
                              batch,
                              nhead,
                              nhead,
                              seqlen,
                              seqlen,
                              hdim,
                              hdim,
                              scale,
                              &lse_ref);

            std::vector<float> lse_host(lse_elems);
            lse_dev.copy_to_host(lse_host.data());

            int lse_reasonable = 0;
            double max_lse_err = 0.0;
            for(int64_t i = 0; i < lse_elems; ++i)
            {
                if(std::isfinite(lse_host[i]) && std::abs(lse_host[i]) < 100.0f)
                    ++lse_reasonable;
                double err  = std::abs(lse_host[i] - lse_ref[i]);
                max_lse_err = std::max(max_lse_err, err);
            }
            std::cout << "  LSE reasonable: " << lse_reasonable << " / " << lse_elems << "\n";
            std::cout << "  LSE max error vs ref: " << std::scientific << max_lse_err << "\n";
            std::cout << "  LSE sample [0..3]: ";
            for(int i = 0; i < std::min<int>(4, lse_elems); ++i)
                std::cout << std::fixed << std::setprecision(4) << lse_host[i] << " ";
            std::cout << "\n";
            passed = (lse_reasonable == lse_elems) && (max_lse_err < 1.0);
        }
        catch(const std::exception& e)
        {
            std::cerr << "  ERROR: " << e.what() << "\n";
        }
        results.push_back({"LSE", passed, time_ms});
    }

    // -----------------------------------------------------------------------
    // Feature C: Dropout
    // -----------------------------------------------------------------------
    {
        std::cout << "\nStep 2c: Dropout (p_drop=0.2)\n";
        const int nhead = 4;

        const int64_t qkv_elems     = static_cast<int64_t>(batch) * nhead * seqlen * hdim;
        const int64_t lse_elems     = static_cast<int64_t>(batch) * nhead * seqlen;
        const int64_t randval_elems = static_cast<int64_t>(batch) * nhead * seqlen * seqlen;

        GpuBuffer<FmhaDataType> q_dev(qkv_elems);
        GpuBuffer<FmhaDataType> k_dev(qkv_elems);
        GpuBuffer<FmhaDataType> v_dev(qkv_elems);
        GpuBuffer<FmhaDataType> o_dev(qkv_elems);
        GpuBuffer<float> lse_dev(lse_elems);
        GpuBuffer<uint8_t> rand_val_dev(randval_elems);

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

        fmha_fwd_traits traits{};
        traits.hdim_q              = hdim;
        traits.hdim_v              = hdim;
        traits.data_type           = "fp16";
        traits.is_group_mode       = false;
        traits.is_v_rowmajor       = true;
        traits.has_logits_soft_cap = false;
        traits.mask_type           = mask_enum::no_mask;
        traits.bias_type           = bias_enum::no_bias;
        traits.has_lse             = true;
        traits.has_dropout         = true;
        traits.qscale_type         = quant_scale_enum::no_scale;

        auto fmha_args                 = make_base_args(q_dev.get(),
                                        k_dev.get(),
                                        v_dev.get(),
                                        o_dev.get(),
                                        batch,
                                        nhead,
                                        nhead,
                                        seqlen,
                                        hdim,
                                        scale);
        fmha_args.lse_ptr              = lse_dev.get();
        fmha_args.rand_val_ptr         = rand_val_dev.get();
        fmha_args.nhead_stride_lse     = seqlen;
        fmha_args.batch_stride_lse     = nhead * seqlen;
        fmha_args.stride_randval       = seqlen;
        fmha_args.nhead_stride_randval = seqlen * seqlen;
        fmha_args.batch_stride_randval = nhead * seqlen * seqlen;
        fmha_args.p_drop               = 0.2f;
        fmha_args.s_randval            = true;
        fmha_args.drop_seed_offset     = std::make_pair(uint64_t(42), uint64_t(0));

        bool passed   = false;
        float time_ms = 0.0f;
        try
        {
            time_ms = dispatcher.run_fwd(traits, fmha_args, nullptr);
            std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";

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
            passed = (nonzero > 0) && (lse_reasonable == lse_elems);
        }
        catch(const std::exception& e)
        {
            std::cerr << "  ERROR: " << e.what() << "\n";
        }
        results.push_back({"Dropout", passed, time_ms});
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "\nStep 3: Summary\n";
    std::cout << "  " << std::setw(16) << "Feature" << " | " << std::setw(10) << "Time(ms)" << " | "
              << std::setw(8) << "Status" << "\n";
    std::cout << "  " << std::string(42, '-') << "\n";

    bool all_passed = true;
    for(const auto& r : results)
    {
        std::cout << "  " << std::setw(16) << r.name << " | " << std::fixed << std::setprecision(4)
                  << std::setw(10) << r.time_ms << " | " << std::setw(8)
                  << (r.passed ? "PASS" : "FAIL") << "\n";
        if(!r.passed)
            all_passed = false;
    }

    print_separator();
    std::cout << "Status: " << (all_passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return all_passed ? 0 : 1;
}
