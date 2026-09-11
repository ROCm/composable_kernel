// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 14: FMHA Benchmark with Validation
//
// Demonstrates:
//   1. Warmup runs to stabilize GPU clocks
//   2. Repeated benchmark runs with statistics (min/avg/max/median)
//   3. Optional CPU reference validation via --verify flag
//
// Usage:
//   ./14_benchmark_validation_fmha
//   ./14_benchmark_validation_fmha --seqlen 256 --batch 4 --repeat 20
//   ./14_benchmark_validation_fmha --verify

#include <hip/hip_runtime.h>
#include <algorithm>
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

using FmhaDataType = ck_tile::fp16_t;

DECL_FMHA_KERNEL_SET(benchmark_fmha_kernels,
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

} // namespace

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 14: FMHA Benchmark + Validation",
                     "Warmup, repeated benchmark, optional verification");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--nhead", "8", "Number of heads");
    args.add_option("--seqlen", "128", "Sequence length (Q and K)");
    args.add_option("--hdim", "128", "Head dimension");
    args.add_option("--warmup", "3", "Warmup iterations");
    args.add_option("--repeat", "10", "Benchmark repetitions");
    args.add_flag("--verify", "Validate against CPU reference");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 2);
    const int nhead            = args.get_int("--nhead", 8);
    const int seqlen           = args.get_int("--seqlen", 128);
    const int hdim             = args.get_int("--hdim", 128);
    const int warmup           = args.get_int("--warmup", 3);
    const int repeat           = args.get_int("--repeat", 10);

    print_header("Example 14: FMHA Benchmark + Validation");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("benchmark_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));

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

    const int64_t q_elems = static_cast<int64_t>(batch) * nhead * seqlen * hdim;
    const int64_t o_elems = q_elems;

    // Step 2: Allocate GPU buffers
    std::cout << "\nStep 2: Allocate GPU Buffers\n";
    std::cout << "  Q/K/V/O: [" << batch << ", " << nhead << ", " << seqlen << ", " << hdim
              << "]\n";

    GpuBuffer<FmhaDataType> q_dev(q_elems);
    GpuBuffer<FmhaDataType> k_dev(q_elems);
    GpuBuffer<FmhaDataType> v_dev(q_elems);
    GpuBuffer<FmhaDataType> o_dev(o_elems);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<FmhaDataType> q_host(q_elems);
    std::vector<FmhaDataType> k_host(q_elems);
    std::vector<FmhaDataType> v_host(q_elems);
    for(auto& x : q_host)
        x = FmhaDataType(dist(rng));
    for(auto& x : k_host)
        x = FmhaDataType(dist(rng));
    for(auto& x : v_host)
        x = FmhaDataType(dist(rng));

    q_dev.copy_from_host(q_host.data());
    k_dev.copy_from_host(k_host.data());
    v_dev.copy_from_host(v_host.data());

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

    FmhaDispatcher dispatcher(&registry);

    // Step 3: Warmup runs
    std::cout << "\nStep 3: Warmup (" << warmup << " iterations)\n";
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 1);
    for(int i = 0; i < warmup; ++i)
    {
        o_dev.zero();
        float t = dispatcher.run_fwd(traits, fmha_args, nullptr);
        std::cout << "  Warmup " << (i + 1) << ": " << std::fixed << std::setprecision(4) << t
                  << " ms\n";
    }

    // Step 4: Benchmark runs
    std::cout << "\nStep 4: Benchmark (" << repeat << " iterations)\n";
    dispatcher.set_timing(0, 1);
    std::vector<float> times;
    times.reserve(repeat);

    for(int i = 0; i < repeat; ++i)
    {
        o_dev.zero();
        float t = dispatcher.run_fwd(traits, fmha_args, nullptr);
        times.push_back(t);
    }

    std::sort(times.begin(), times.end());
    float t_min = times.front();
    float t_max = times.back();
    float t_avg = std::accumulate(times.begin(), times.end(), 0.0f) / static_cast<float>(repeat);
    float t_med =
        (repeat % 2 == 0) ? (times[repeat / 2 - 1] + times[repeat / 2]) / 2.0f : times[repeat / 2];

    auto problem = FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_args), gfx_arch);
    double ops   = static_cast<double>(problem.num_ops());
    double tflops_min = ops / (t_max * 1e-3) / 1e12;
    double tflops_max = ops / (t_min * 1e-3) / 1e12;
    double tflops_avg = ops / (t_avg * 1e-3) / 1e12;
    double tflops_med = ops / (t_med * 1e-3) / 1e12;

    std::cout << "\n  " << std::setw(10) << "Metric" << " | " << std::setw(12) << "Time(ms)"
              << " | " << std::setw(12) << "TFLOPS" << "\n";
    std::cout << "  " << std::string(40, '-') << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << std::setw(10) << "Min" << " | " << std::setw(12) << t_min << " | "
              << std::setprecision(2) << std::setw(12) << tflops_max << "\n";
    std::cout << std::setprecision(4);
    std::cout << "  " << std::setw(10) << "Avg" << " | " << std::setw(12) << t_avg << " | "
              << std::setprecision(2) << std::setw(12) << tflops_avg << "\n";
    std::cout << std::setprecision(4);
    std::cout << "  " << std::setw(10) << "Median" << " | " << std::setw(12) << t_med << " | "
              << std::setprecision(2) << std::setw(12) << tflops_med << "\n";
    std::cout << std::setprecision(4);
    std::cout << "  " << std::setw(10) << "Max" << " | " << std::setw(12) << t_max << " | "
              << std::setprecision(2) << std::setw(12) << tflops_min << "\n";

    bool passed = true;

    // Step 5: Optional validation
    if(args.has("--verify"))
    {
        std::cout << "\nStep 5: CPU Reference Validation\n";

        std::vector<FmhaDataType> o_host(o_elems);
        o_dev.copy_to_host(o_host.data());

        std::vector<float> q_f32(q_elems), k_f32(q_elems), v_f32(q_elems), o_ref(o_elems, 0.0f);
        for(int64_t i = 0; i < q_elems; ++i)
            q_f32[i] = static_cast<float>(q_host[i]);
        for(int64_t i = 0; i < q_elems; ++i)
            k_f32[i] = static_cast<float>(k_host[i]);
        for(int64_t i = 0; i < q_elems; ++i)
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
    else
    {
        std::vector<FmhaDataType> o_host(o_elems);
        o_dev.copy_to_host(o_host.data());
        int nonzero = 0;
        for(int64_t i = 0; i < o_elems; ++i)
        {
            if(static_cast<float>(o_host[i]) != 0.0f)
                ++nonzero;
        }
        std::cout << "\n  Sanity: " << nonzero << " / " << o_elems << " non-zero outputs\n";
        passed = (nonzero > 0);
    }

    print_separator();
    std::cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return passed ? 0 : 1;
}
