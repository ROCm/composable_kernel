// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 26: Multiple Data Types and Head Dimensions with GPU Execution
//
// Demonstrates:
//   1. Declare bf16 hdim=128, fp16 hdim=64, and fp16 hdim=128 kernels
//   2. Run each variant on GPU with appropriate buffer types
//   3. Validate with different tolerances: fp16 (rtol=1e-3), bf16 (rtol=1e-2)
//   4. Mention fp32, fp8bf16, fp8fp32, hdim 256, asymmetric hdim as planning
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

DECL_FMHA_KERNEL_SET(dtypes_hdims_kernels,

                     // bf16 hdim=128
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("bf16")
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

                         // fp16 hdim=64
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(64)
                                  .mask("no")
                                  .bias("no")
                                  .lse(false)
                                  .dropout(false)
                                  .qscale("no"),
                              FmhaAlgorithm()
                                  .tile_m0(128)
                                  .tile_n0(64)
                                  .tile_k0(32)
                                  .tile_n1(64)
                                  .tile_k1(32)
                                  .tile_k0max(64)
                                  .wave_m0(4)
                                  .wave_n0(1)
                                  .wave_k0(1)
                                  .wave_m1(4)
                                  .wave_n1(1)
                                  .wave_k1(1)
                                  .warp_m0(32)
                                  .warp_n0(32)
                                  .warp_k0(16)
                                  .warp_m1(16)
                                  .warp_n1(16)
                                  .warp_k1(16)
                                  .pipeline("qr_async")
                                  .padding(true, true, true, true)
                                  .alignments(64, 64)
                                  .selection_rank(0),
                              "gfx950")

                         // fp16 hdim=128 (reference baseline)
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

using Fp16Type = ck_tile::fp16_t;
using Bf16Type = ck_tile::bf16_t;

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

struct VariantResult
{
    std::string label;
    float time_ms;
    double tflops;
    double max_abs_err;
    double max_rel_err;
    int errors;
    bool passed;
};

template <typename DataType>
fmha_fwd_args make_fwd_args(GpuBuffer<DataType>& q_dev,
                            GpuBuffer<DataType>& k_dev,
                            GpuBuffer<DataType>& v_dev,
                            GpuBuffer<DataType>& o_dev,
                            int batch,
                            int nhead,
                            int seqlen,
                            int hdim,
                            float scale)
{
    fmha_fwd_args a{};
    a.q_ptr = q_dev.get();
    a.k_ptr = k_dev.get();
    a.v_ptr = v_dev.get();
    a.o_ptr = o_dev.get();

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
    a.nhead_q         = nhead;
    a.nhead_k         = nhead;
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

    a.batch_stride_q         = nhead * seqlen * hdim;
    a.batch_stride_k         = nhead * seqlen * hdim;
    a.batch_stride_v         = nhead * seqlen * hdim;
    a.batch_stride_bias      = 0;
    a.batch_stride_randval   = 0;
    a.batch_stride_lse       = 0;
    a.batch_stride_o         = nhead * seqlen * hdim;
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

template <typename DataType>
VariantResult run_variant(FmhaDispatcher& dispatcher,
                          const std::string& label,
                          const std::string& dtype_str,
                          int batch,
                          int nhead,
                          int seqlen,
                          int hdim,
                          double rtol,
                          double atol,
                          const std::string& gfx_arch)
{
    VariantResult result{};
    result.label = label;

    const float scale   = 1.0f / std::sqrt(static_cast<float>(hdim));
    const int64_t elems = static_cast<int64_t>(batch) * nhead * seqlen * hdim;

    fmha_fwd_traits traits{};
    traits.hdim_q              = hdim;
    traits.hdim_v              = hdim;
    traits.data_type           = dtype_str;
    traits.is_group_mode       = false;
    traits.is_v_rowmajor       = true;
    traits.has_logits_soft_cap = false;
    traits.mask_type           = mask_enum::no_mask;
    traits.bias_type           = bias_enum::no_bias;
    traits.has_lse             = false;
    traits.has_dropout         = false;
    traits.qscale_type         = quant_scale_enum::no_scale;

    GpuBuffer<DataType> q_dev(elems);
    GpuBuffer<DataType> k_dev(elems);
    GpuBuffer<DataType> v_dev(elems);
    GpuBuffer<DataType> o_dev(elems);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<DataType> q_host(elems);
    std::vector<DataType> k_host(elems);
    std::vector<DataType> v_host(elems);
    for(auto& x : q_host)
        x = DataType(dist(rng));
    for(auto& x : k_host)
        x = DataType(dist(rng));
    for(auto& x : v_host)
        x = DataType(dist(rng));

    q_dev.copy_from_host(q_host.data());
    k_dev.copy_from_host(k_host.data());
    v_dev.copy_from_host(v_host.data());
    o_dev.zero();

    auto fwd_args = make_fwd_args(q_dev, k_dev, v_dev, o_dev, batch, nhead, seqlen, hdim, scale);

    try
    {
        result.time_ms = dispatcher.run_fwd(traits, fwd_args, nullptr);
    }
    catch(const std::exception& e)
    {
        std::cerr << "  ERROR [" << label << "]: " << e.what() << "\n";
        result.passed = false;
        return result;
    }

    auto problem  = FmhaProblem::from_invocation(FmhaInvocation::make(traits, fwd_args), gfx_arch);
    result.tflops = static_cast<double>(problem.num_ops()) / (result.time_ms * 1e-3) / 1e12;

    std::vector<DataType> o_host(elems);
    o_dev.copy_to_host(o_host.data());

    std::vector<float> q_f32(elems), k_f32(elems), v_f32(elems), o_ref(elems, 0.0f);
    for(int64_t i = 0; i < elems; ++i)
        q_f32[i] = static_cast<float>(q_host[i]);
    for(int64_t i = 0; i < elems; ++i)
        k_f32[i] = static_cast<float>(k_host[i]);
    for(int64_t i = 0; i < elems; ++i)
        v_f32[i] = static_cast<float>(v_host[i]);

    cpu_attention_fwd(q_f32, k_f32, v_f32, o_ref, batch, nhead, seqlen, seqlen, hdim, hdim, scale);

    result.max_abs_err = 0.0;
    result.max_rel_err = 0.0;
    result.errors      = 0;

    for(int64_t i = 0; i < elems; ++i)
    {
        float gpu_val      = static_cast<float>(o_host[i]);
        float ref_val      = o_ref[i];
        double abs_err     = std::abs(gpu_val - ref_val);
        double rel_err     = abs_err / (std::abs(ref_val) + 1e-6);
        result.max_abs_err = std::max(result.max_abs_err, abs_err);
        result.max_rel_err = std::max(result.max_rel_err, rel_err);
        if(abs_err > atol + rtol * std::abs(ref_val))
            ++result.errors;
    }

    result.passed = (result.errors == 0);
    return result;
}

} // namespace

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 26: Dtypes & Hdims FMHA",
                     "FMHA with multiple data types and head dimensions");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--nhead", "4", "Number of heads");
    args.add_option("--seqlen", "64", "Sequence length (Q and K)");
    args.add_flag("--validate", "Validate against CPU reference");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 2);
    const int nhead            = args.get_int("--nhead", 4);
    const int seqlen           = args.get_int("--seqlen", 64);

    print_header("Example 26: Multiple Data Types & Head Dimensions");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("dtypes_hdims");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 3);

    // =========================================================================
    // Step 2: Run variants on GPU
    // =========================================================================
    std::cout << "\nStep 2: Run Variants\n";

    // fp16 hdim=128 (reference baseline)
    std::cout << "\n  --- fp16 hdim=128 (reference) ---\n";
    auto r_fp16_h128 = run_variant<Fp16Type>(dispatcher,
                                             "fp16_h128",
                                             "fp16",
                                             batch,
                                             nhead,
                                             seqlen,
                                             128,
                                             /*rtol=*/1e-3,
                                             /*atol=*/1e-3,
                                             gfx_arch);

    // bf16 hdim=128 (wider tolerance due to reduced precision)
    std::cout << "\n  --- bf16 hdim=128 ---\n";
    auto r_bf16_h128 = run_variant<Bf16Type>(dispatcher,
                                             "bf16_h128",
                                             "bf16",
                                             batch,
                                             nhead,
                                             seqlen,
                                             128,
                                             /*rtol=*/1e-2,
                                             /*atol=*/1e-2,
                                             gfx_arch);

    // fp16 hdim=64 (smaller buffers)
    std::cout << "\n  --- fp16 hdim=64 ---\n";
    auto r_fp16_h64 = run_variant<Fp16Type>(dispatcher,
                                            "fp16_h64",
                                            "fp16",
                                            batch,
                                            nhead,
                                            seqlen,
                                            64,
                                            /*rtol=*/1e-3,
                                            /*atol=*/1e-3,
                                            gfx_arch);

    // =========================================================================
    // Step 3: Results Summary
    // =========================================================================
    std::cout << "\nStep 3: Results Summary\n\n";

    std::cout << "  " << std::setw(14) << "Variant" << " | " << std::setw(10) << "Time(ms)" << " | "
              << std::setw(10) << "TFLOPS" << " | " << std::setw(10) << "MaxAbsErr" << " | "
              << std::setw(10) << "MaxRelErr" << " | " << std::setw(8) << "Errors" << " | "
              << std::setw(6) << "Status" << "\n";
    std::cout << "  " << std::string(82, '-') << "\n";

    auto print_row = [](const VariantResult& r) {
        std::cout << std::fixed;
        std::cout << "  " << std::setw(14) << r.label << " | " << std::setprecision(4)
                  << std::setw(10) << r.time_ms << " | " << std::setprecision(2) << std::setw(10)
                  << r.tflops << " | " << std::scientific << std::setw(10) << r.max_abs_err << " | "
                  << std::setw(10) << r.max_rel_err << " | " << std::fixed << std::setw(8)
                  << r.errors << " | " << std::setw(6) << (r.passed ? "PASS" : "FAIL") << "\n";
    };

    print_row(r_fp16_h128);
    print_row(r_bf16_h128);
    print_row(r_fp16_h64);

    // =========================================================================
    // Step 4: Tolerance Notes
    // =========================================================================
    std::cout << "\nStep 4: Tolerance Notes\n";
    std::cout << "  fp16 validation: rtol=1e-3, atol=1e-3 (higher precision)\n";
    std::cout << "  bf16 validation: rtol=1e-2, atol=1e-2 (wider tolerance for bfloat16)\n";
    std::cout << "\n  Additional dtype/hdim combinations (planning-level declarations):\n";
    std::cout << "    fp32:           .dtype(\"fp32\") - full single precision\n";
    std::cout << "    fp8bf16:        .dtype(\"fp8bf16\") - fp8 compute, bf16 output\n";
    std::cout << "    fp8fp32:        .dtype(\"fp8fp32\") - fp8 compute, fp32 output\n";
    std::cout << "    hdim 256:       .hdim(256), tile(128,128,32,256,32,256)\n";
    std::cout << "    asymmetric:     .hdim_q(128), .hdim_v(64) - different Q/V dims\n";

    bool all_passed = r_fp16_h128.passed && r_bf16_h128.passed && r_fp16_h64.passed;

    print_separator();
    std::cout << "Status: " << (all_passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return all_passed ? 0 : 1;
}
