// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 17: FMHA Autofill and Autocorrect
//
// Demonstrates three DECL_FMHA_KERNEL_SET patterns:
//   1. AUTOFILL: Minimal specification -- only family/dtype/hdim/pipeline/tile
//      are provided; wave/warp use defaults from FmhaAlgorithm constructor
//   2. AUTOCORRECT: Intentionally non-standard wave config that still works
//      because FmhaAlgorithm auto_fill() corrects missing tile_n1/tile_k1
//   3. FULL: All parameters explicitly specified (reference)
//
// Each is registered, planned, run on GPU, and validated.
//
// Usage:
//   ./17_autofill_autocorrect_fmha
//   ./17_autofill_autocorrect_fmha --arch gfx942

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

using FmhaDataType = ck_tile::fp16_t;

// Pattern 1: AUTOFILL -- minimal specification, defaults for wave/warp
DECL_FMHA_KERNEL_SET(autofill_kernels,
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
                              .pipeline("qr_async")
                              .padding(true, true, true, true),
                          "gfx950"));

// Pattern 2: AUTOCORRECT -- tile_n1/tile_k1 set to 0, auto_fill() corrects them
DECL_FMHA_KERNEL_SET(autocorrect_kernels,
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
                              .tile_n1(0)
                              .tile_k1(0)
                              .tile_k0max(0)
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
                              .padding(true, true, true, true),
                          "gfx950"));

// Pattern 3: FULL -- every parameter explicitly specified
DECL_FMHA_KERNEL_SET(full_spec_kernels,
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

struct KernelTestCase
{
    std::string name;
    std::string kernel_set_name;
};

} // namespace

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 17: FMHA Autofill & Autocorrect",
                     "Three DECL_FMHA_KERNEL_SET patterns compared");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--nhead", "8", "Number of heads");
    args.add_option("--seqlen", "128", "Sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 2);
    const int nhead            = args.get_int("--nhead", 8);
    const int seqlen           = args.get_int("--seqlen", 128);
    const int hdim             = args.get_int("--hdim", 128);

    print_header("Example 17: FMHA Autofill & Autocorrect");

    // Step 1: Show registered kernel sets
    std::cout << "\nStep 1: Registered Kernel Sets\n";
    FmhaKernelSetRegistry::instance().print();

    const KernelTestCase cases[] = {
        {"AUTOFILL    (minimal spec, wave/warp defaults)", "autofill_kernels"},
        {"AUTOCORRECT (tile_n1/k1=0, auto_fill corrects)", "autocorrect_kernels"},
        {"FULL        (all params explicit)", "full_spec_kernels"},
    };

    // Prepare input data (shared across all tests)
    const float scale   = 1.0f / std::sqrt(static_cast<float>(hdim));
    const int64_t elems = static_cast<int64_t>(batch) * nhead * seqlen * hdim;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<FmhaDataType> q_host(elems), k_host(elems), v_host(elems);
    for(auto& x : q_host)
        x = FmhaDataType(dist(rng));
    for(auto& x : k_host)
        x = FmhaDataType(dist(rng));
    for(auto& x : v_host)
        x = FmhaDataType(dist(rng));

    // CPU reference
    std::vector<float> q_f32(elems), k_f32(elems), v_f32(elems), o_ref(elems, 0.0f);
    for(int64_t i = 0; i < elems; ++i)
        q_f32[i] = static_cast<float>(q_host[i]);
    for(int64_t i = 0; i < elems; ++i)
        k_f32[i] = static_cast<float>(k_host[i]);
    for(int64_t i = 0; i < elems; ++i)
        v_f32[i] = static_cast<float>(v_host[i]);
    cpu_attention_fwd(q_f32, k_f32, v_f32, o_ref, batch, nhead, seqlen, seqlen, hdim, hdim, scale);

    int total_pass        = 0;
    const int total_cases = sizeof(cases) / sizeof(cases[0]);

    for(int ci = 0; ci < total_cases; ++ci)
    {
        const auto& tc = cases[ci];
        std::cout << "\nStep " << (ci + 2) << ": " << tc.name << "\n";

        // Register from the named kernel set
        FmhaRegistry registry;
        registry.set_name(tc.kernel_set_name);
        REGISTER_GENERATED_KERNELS(registry, gfx_arch);
        std::cout << "  Registered " << registry.size() << " kernel(s)\n";

        if(registry.size() == 0)
        {
            std::cout << "  SKIP: no kernels registered\n";
            continue;
        }

        FmhaDispatcher dispatcher(&registry);
        dispatcher.set_benchmarking(true);
        dispatcher.set_timing(1, 3);

        // Allocate GPU buffers
        GpuBuffer<FmhaDataType> q_dev(elems);
        GpuBuffer<FmhaDataType> k_dev(elems);
        GpuBuffer<FmhaDataType> v_dev(elems);
        GpuBuffer<FmhaDataType> o_dev(elems);

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

        fmha_fwd_args fmha_args{};
        fmha_args.q_ptr = q_dev.get();
        fmha_args.k_ptr = k_dev.get();
        fmha_args.v_ptr = v_dev.get();
        fmha_args.o_ptr = o_dev.get();

        fmha_args.bias_ptr                   = nullptr;
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

        fmha_args.stride_q       = hdim;
        fmha_args.stride_k       = hdim;
        fmha_args.stride_v       = hdim;
        fmha_args.stride_bias    = 0;
        fmha_args.stride_randval = 0;
        fmha_args.stride_o       = hdim;

        fmha_args.nhead_stride_q         = seqlen * hdim;
        fmha_args.nhead_stride_k         = seqlen * hdim;
        fmha_args.nhead_stride_v         = seqlen * hdim;
        fmha_args.nhead_stride_bias      = 0;
        fmha_args.nhead_stride_randval   = 0;
        fmha_args.nhead_stride_lse       = 0;
        fmha_args.nhead_stride_o         = seqlen * hdim;
        fmha_args.nhead_stride_q_descale = 0;
        fmha_args.nhead_stride_k_descale = 0;
        fmha_args.nhead_stride_v_descale = 0;

        fmha_args.batch_stride_q         = nhead * seqlen * hdim;
        fmha_args.batch_stride_k         = nhead * seqlen * hdim;
        fmha_args.batch_stride_v         = nhead * seqlen * hdim;
        fmha_args.batch_stride_bias      = 0;
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

        try
        {
            float time_ms = dispatcher.run_fwd(traits, fmha_args, nullptr);

            auto problem =
                FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_args), gfx_arch);
            double tflops = static_cast<double>(problem.num_ops()) / (time_ms * 1e-3) / 1e12;

            // Validate
            std::vector<FmhaDataType> o_host(elems);
            o_dev.copy_to_host(o_host.data());

            double max_abs_err = 0.0;
            int errors         = 0;
            for(int64_t i = 0; i < elems; ++i)
            {
                double abs_err = std::abs(static_cast<float>(o_host[i]) - o_ref[i]);
                max_abs_err    = std::max(max_abs_err, abs_err);
                if(abs_err > 1e-2 + 1e-2 * std::abs(o_ref[i]))
                    ++errors;
            }

            bool ok = (errors == 0);
            std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms"
                      << "  TFLOPS: " << std::setprecision(2) << tflops
                      << "  MaxErr: " << std::scientific << max_abs_err << "  "
                      << (ok ? "PASS" : "FAIL") << "\n";
            if(ok)
                ++total_pass;
        }
        catch(const std::exception& e)
        {
            std::cerr << "  ERROR: " << e.what() << "\n";
        }
    }

    // Summary
    print_separator();
    std::cout << "Results: " << total_pass << "/" << total_cases << " patterns passed\n";
    std::cout << "Patterns:\n";
    std::cout << "  1. AUTOFILL:    Only tile + pipeline specified; wave/warp use defaults\n";
    std::cout << "  2. AUTOCORRECT: tile_n1/k1/k0max=0 -> auto_fill() infers from tile_n0/k0\n";
    std::cout << "  3. FULL:        Every parameter explicit (reference configuration)\n";
    std::cout << "Status: " << (total_pass == total_cases ? "PASS" : "FAIL") << "\n";
    print_separator();

    return (total_pass == total_cases) ? 0 : 1;
}
