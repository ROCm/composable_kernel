// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 20: GPU FMHA Forward with Bias Types
//
// Demonstrates three bias variants with GPU execution:
//   1. No bias (standard attention)
//   2. Elementwise bias (arbitrary bias matrix added to scores)
//   3. ALiBi (Attention with Linear Biases -- slope-based positional encoding)
//
// Validates each variant against a CPU reference.

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

DECL_FMHA_KERNEL_SET(bias_fmha_kernels,
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
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("bias")
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
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("alibi")
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

// bias_type: 0=none, 1=elementwise, 2=alibi
// bias_buf layout: elementwise [1, nhead, seqlen_q, seqlen_k], alibi [1, nhead] slopes
void cpu_attention_fwd_biased(const std::vector<float>& Q,
                              const std::vector<float>& K,
                              const std::vector<float>& V,
                              std::vector<float>& O,
                              int batch,
                              int nhead,
                              int seqlen_q,
                              int seqlen_k,
                              int hdim_q,
                              int hdim_v,
                              float scale,
                              int bias_type,
                              const std::vector<float>& bias_buf)
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
                    float s = dot * scale;

                    if(bias_type == 1)
                    {
                        int bias_idx = (h * seqlen_q + sq) * seqlen_k + sk;
                        s += bias_buf[bias_idx];
                    }
                    else if(bias_type == 2)
                    {
                        float slope = bias_buf[h];
                        s += slope * static_cast<float>(sk - sq);
                    }

                    scores[sk] = s;
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
    ExampleArgs args("Example 20: FMHA with Bias (GPU)", "FMHA bias variants on GPU");
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

    print_header("Example 20: FMHA with Bias (GPU)");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("bias_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 3);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));

    // Allocate Q, K, V GPU buffers (shared across all bias tests)
    const int64_t q_elems = static_cast<int64_t>(batch) * nhead * seqlen * hdim;
    const int64_t k_elems = q_elems;
    const int64_t v_elems = q_elems;
    const int64_t o_elems = q_elems;

    std::cout << "\nStep 2: Allocate GPU Buffers\n";
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

    // Convert to f32 for CPU reference
    std::vector<float> q_f32(q_elems), k_f32(k_elems), v_f32(v_elems);
    for(int64_t i = 0; i < q_elems; ++i)
        q_f32[i] = static_cast<float>(q_host[i]);
    for(int64_t i = 0; i < k_elems; ++i)
        k_f32[i] = static_cast<float>(k_host[i]);
    for(int64_t i = 0; i < v_elems; ++i)
        v_f32[i] = static_cast<float>(v_host[i]);

    // Prepare elementwise bias buffer: [1, nhead, seqlen, seqlen] with small values
    const int64_t elem_bias_elems = static_cast<int64_t>(nhead) * seqlen * seqlen;
    std::vector<float> elem_bias_host(elem_bias_elems);
    std::uniform_real_distribution<float> bias_dist(-0.1f, 0.1f);
    for(auto& x : elem_bias_host)
        x = bias_dist(rng);

    GpuBuffer<float> elem_bias_dev(elem_bias_elems);
    elem_bias_dev.copy_from_host(elem_bias_host.data());

    // Prepare ALiBi slopes buffer: [nhead] with geometric slopes
    std::vector<float> alibi_slopes_host(nhead);
    for(int h = 0; h < nhead; ++h)
    {
        alibi_slopes_host[h] = -std::pow(2.0f, -(8.0f * (h + 1) / nhead));
    }

    GpuBuffer<float> alibi_slopes_dev(nhead);
    alibi_slopes_dev.copy_from_host(alibi_slopes_host.data());

    // Test each bias type
    struct BiasTest
    {
        const char* name;
        int bias_type_int;
        bias_enum bias_type;
        void* bias_ptr;
        int stride_bias;
        int nhead_stride_bias;
        int batch_stride_bias;
    };

    BiasTest tests[] = {
        {"no_bias", 0, bias_enum::no_bias, nullptr, 0, 0, 0},
        {"elementwise_bias",
         1,
         bias_enum::elementwise_bias,
         elem_bias_dev.get(),
         seqlen,
         seqlen * seqlen,
         0},
        {"alibi", 2, bias_enum::alibi, alibi_slopes_dev.get(), 0, 1, 0},
    };

    bool all_passed = true;

    for(const auto& test : tests)
    {
        std::cout << "\nStep 3: Run FMHA Forward [" << test.name << "]\n";

        fmha_fwd_traits traits{};
        traits.hdim_q              = hdim;
        traits.hdim_v              = hdim;
        traits.data_type           = "fp16";
        traits.is_group_mode       = false;
        traits.is_v_rowmajor       = true;
        traits.has_logits_soft_cap = false;
        traits.mask_type           = mask_enum::no_mask;
        traits.bias_type           = test.bias_type;
        traits.has_lse             = false;
        traits.has_dropout         = false;
        traits.qscale_type         = quant_scale_enum::no_scale;

        o_dev.zero();

        fmha_fwd_args fmha_args{};
        fmha_args.q_ptr = q_dev.get();
        fmha_args.k_ptr = k_dev.get();
        fmha_args.v_ptr = v_dev.get();
        fmha_args.o_ptr = o_dev.get();

        fmha_args.bias_ptr                   = test.bias_ptr;
        fmha_args.q_descale_ptr              = nullptr;
        fmha_args.k_descale_ptr              = nullptr;
        fmha_args.v_descale_ptr              = nullptr;
        fmha_args.rand_val_ptr               = nullptr;
        fmha_args.lse_ptr                    = nullptr;
        fmha_args.sink_ptr                   = nullptr;
        fmha_args.block_scale_seqstart_q_ptr = nullptr;
        fmha_args.block_scale_seqstart_k_ptr = nullptr;

        fmha_args.seqlen_q        = seqlen;
        fmha_args.seqlen_k        = seqlen;
        fmha_args.batch           = batch;
        fmha_args.max_seqlen_q    = seqlen;
        fmha_args.hdim_q          = hdim;
        fmha_args.hdim_v          = hdim;
        fmha_args.nhead_q         = nhead;
        fmha_args.nhead_k         = nhead;
        fmha_args.scale_s         = scale;
        fmha_args.logits_soft_cap = 0.0f;

        // bhsd layout strides
        fmha_args.stride_q       = hdim;
        fmha_args.stride_k       = hdim;
        fmha_args.stride_v       = hdim;
        fmha_args.stride_bias    = test.stride_bias;
        fmha_args.stride_randval = 0;
        fmha_args.stride_o       = hdim;

        fmha_args.nhead_stride_q         = seqlen * hdim;
        fmha_args.nhead_stride_k         = seqlen * hdim;
        fmha_args.nhead_stride_v         = seqlen * hdim;
        fmha_args.nhead_stride_bias      = test.nhead_stride_bias;
        fmha_args.nhead_stride_randval   = 0;
        fmha_args.nhead_stride_lse       = 0;
        fmha_args.nhead_stride_o         = seqlen * hdim;
        fmha_args.nhead_stride_q_descale = 0;
        fmha_args.nhead_stride_k_descale = 0;
        fmha_args.nhead_stride_v_descale = 0;

        fmha_args.batch_stride_q         = nhead * seqlen * hdim;
        fmha_args.batch_stride_k         = nhead * seqlen * hdim;
        fmha_args.batch_stride_v         = nhead * seqlen * hdim;
        fmha_args.batch_stride_bias      = test.batch_stride_bias;
        fmha_args.batch_stride_randval   = 0;
        fmha_args.batch_stride_lse       = 0;
        fmha_args.batch_stride_o         = nhead * seqlen * hdim;
        fmha_args.batch_stride_q_descale = 0;
        fmha_args.batch_stride_k_descale = 0;
        fmha_args.batch_stride_v_descale = 0;

        fmha_args.window_size_left    = -1;
        fmha_args.window_size_right   = -1;
        fmha_args.sink_size           = 0;
        fmha_args.mask_type           = 0;
        fmha_args.min_seqlen_q        = 0;
        fmha_args.p_drop              = 0.0f;
        fmha_args.s_randval           = false;
        fmha_args.drop_seed_offset    = std::make_pair(uint64_t(0), uint64_t(0));
        fmha_args.block_scale_size_q  = 0;
        fmha_args.block_scale_size_kv = 0;

        float time_ms = 0.0f;
        try
        {
            time_ms = dispatcher.run_fwd(traits, fmha_args, nullptr);
        }
        catch(const std::exception& e)
        {
            std::cerr << "  ERROR [" << test.name << "]: " << e.what() << "\n";
            all_passed = false;
            continue;
        }

        auto problem =
            FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_args), gfx_arch);
        double tflops = static_cast<double>(problem.num_ops()) / (time_ms * 1e-3) / 1e12;

        std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
        std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";

        // Validate
        std::vector<FmhaDataType> o_host(o_elems);
        o_dev.copy_to_host(o_host.data());

        int nonzero = 0;
        for(int64_t i = 0; i < o_elems; ++i)
        {
            if(static_cast<float>(o_host[i]) != 0.0f)
                ++nonzero;
        }
        std::cout << "  Non-zero outputs: " << nonzero << " / " << o_elems << "\n";

        if(nonzero == 0)
            all_passed = false;

        if(args.has("--validate"))
        {
            std::vector<float> o_ref(o_elems, 0.0f);

            if(test.bias_type_int == 0)
            {
                cpu_attention_fwd(
                    q_f32, k_f32, v_f32, o_ref, batch, nhead, seqlen, seqlen, hdim, hdim, scale);
            }
            else
            {
                const std::vector<float>& bias_ref =
                    (test.bias_type_int == 1) ? elem_bias_host : alibi_slopes_host;
                cpu_attention_fwd_biased(q_f32,
                                         k_f32,
                                         v_f32,
                                         o_ref,
                                         batch,
                                         nhead,
                                         seqlen,
                                         seqlen,
                                         hdim,
                                         hdim,
                                         scale,
                                         test.bias_type_int,
                                         bias_ref);
            }

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
            if(errors > 0)
                all_passed = false;
        }
    }

    print_separator();
    std::cout << "Status: " << (all_passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return all_passed ? 0 : 1;
}
