// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 30: FMHA Backward Benchmark
//
// Demonstrates:
//   1. Forward kernel for benchmark (with LSE for backward planning)
//   2. Multiple problem sizes: sweep batch x seqlen
//   3. GPU forward execution for each size with timing
//   4. Backward plan for each size
//   5. Summary table: Batch | SeqLen | Fwd(ms) | BwdPlan | FwdTFLOPS
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

DECL_FMHA_KERNEL_SET(bwd_bench_fmha_kernels,
                     // Forward: basic fp16 with LSE for backward
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
                          "gfx950")

                         // Backward stage 1: dot(dO, O)
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

                         // Backward stage 2: dQ, dK, dV
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

                         // Backward stage 3: convert dQ
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

struct BenchResult
{
    int batch;
    int seqlen;
    float fwd_ms;
    double fwd_tflops;
    int bwd_stages;
    bool bwd_valid;
    bool fwd_passed;
};

} // namespace

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 30: FMHA Backward Benchmark",
                     "Sweep batch x seqlen, forward GPU + backward plan");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--nhead", "8", "Number of heads");
    args.add_option("--hdim", "128", "Head dimension");
    args.add_option("--warmup", "2", "Warmup iterations per size");
    args.add_option("--repeat", "3", "Benchmark repetitions per size");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int nhead            = args.get_int("--nhead", 8);
    const int hdim             = args.get_int("--hdim", 128);
    const int warmup           = args.get_int("--warmup", 2);
    const int repeat           = args.get_int("--repeat", 3);
    const float scale          = 1.0f / std::sqrt(static_cast<float>(hdim));

    print_header("Example 30: FMHA Backward Benchmark");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("bwd_bench_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);

    // Problem sizes to sweep
    struct ProblemSize
    {
        int batch;
        int seqlen;
    };

    ProblemSize sizes[] = {
        {8, 128},
        {4, 256},
        {2, 512},
        {1, 1024},
        {1, 2048},
        {1, 4096},
    };

    std::vector<BenchResult> results;

    // Step 2: Sweep problem sizes
    std::cout << "\nStep 2: Sweep Problem Sizes\n";

    for(const auto& sz : sizes)
    {
        std::cout << "\n  --- batch=" << sz.batch << ", seqlen=" << sz.seqlen << " ---\n";

        const int64_t qkv_elems = static_cast<int64_t>(sz.batch) * nhead * sz.seqlen * hdim;
        const int64_t lse_elems = static_cast<int64_t>(sz.batch) * nhead * sz.seqlen;

        BenchResult res{};
        res.batch  = sz.batch;
        res.seqlen = sz.seqlen;

        // Allocate buffers
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

        // Forward traits/args
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

        fwd_args.seqlen_q        = sz.seqlen;
        fwd_args.seqlen_k        = sz.seqlen;
        fwd_args.batch           = sz.batch;
        fwd_args.max_seqlen_q    = sz.seqlen;
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

        fwd_args.nhead_stride_q         = sz.seqlen * hdim;
        fwd_args.nhead_stride_k         = sz.seqlen * hdim;
        fwd_args.nhead_stride_v         = sz.seqlen * hdim;
        fwd_args.nhead_stride_bias      = 0;
        fwd_args.nhead_stride_randval   = 0;
        fwd_args.nhead_stride_lse       = sz.seqlen;
        fwd_args.nhead_stride_o         = sz.seqlen * hdim;
        fwd_args.nhead_stride_q_descale = 0;
        fwd_args.nhead_stride_k_descale = 0;
        fwd_args.nhead_stride_v_descale = 0;

        fwd_args.batch_stride_q         = nhead * sz.seqlen * hdim;
        fwd_args.batch_stride_k         = nhead * sz.seqlen * hdim;
        fwd_args.batch_stride_v         = nhead * sz.seqlen * hdim;
        fwd_args.batch_stride_bias      = 0;
        fwd_args.batch_stride_randval   = 0;
        fwd_args.batch_stride_lse       = nhead * sz.seqlen;
        fwd_args.batch_stride_o         = nhead * sz.seqlen * hdim;
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

        // Warmup
        dispatcher.set_benchmarking(true);
        dispatcher.set_timing(1, 1);
        try
        {
            for(int w = 0; w < warmup; ++w)
            {
                o_dev.zero();
                lse_dev.zero();
                dispatcher.run_fwd(fwd_traits, fwd_args, nullptr);
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << "  Warmup ERROR: " << e.what() << "\n";
            res.fwd_passed = false;
            results.push_back(res);
            continue;
        }

        // Benchmark
        dispatcher.set_timing(0, 1);
        float total_ms = 0.0f;
        bool ok        = true;
        for(int r = 0; r < repeat; ++r)
        {
            o_dev.zero();
            lse_dev.zero();
            try
            {
                total_ms += dispatcher.run_fwd(fwd_traits, fwd_args, nullptr);
            }
            catch(const std::exception& e)
            {
                std::cerr << "  Bench ERROR: " << e.what() << "\n";
                ok = false;
                break;
            }
        }

        if(ok)
        {
            res.fwd_ms = total_ms / static_cast<float>(repeat);

            auto problem =
                FmhaProblem::from_invocation(FmhaInvocation::make(fwd_traits, fwd_args), gfx_arch);
            res.fwd_tflops = static_cast<double>(problem.num_ops()) / (res.fwd_ms * 1e-3) / 1e12;

            // Sanity check output
            std::vector<FmhaDataType> o_host(qkv_elems);
            o_dev.copy_to_host(o_host.data());
            int nonzero = 0;
            for(int64_t i = 0; i < qkv_elems; ++i)
            {
                if(static_cast<float>(o_host[i]) != 0.0f)
                    ++nonzero;
            }
            res.fwd_passed = (nonzero > 0);
        }
        else
        {
            res.fwd_passed = false;
        }

        // Backward plan for this size
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
        bwd_args.batch        = sz.batch;
        bwd_args.seqlen_q     = sz.seqlen;
        bwd_args.seqlen_k     = sz.seqlen;
        bwd_args.max_seqlen_q = sz.seqlen;
        bwd_args.max_seqlen_k = sz.seqlen;
        bwd_args.hdim_q       = hdim;
        bwd_args.hdim_v       = hdim;
        bwd_args.nhead_q      = nhead;
        bwd_args.nhead_k      = nhead;

        auto bwd_plan = dispatcher.plan(
            FmhaProblem::from_invocation(FmhaInvocation::make(bwd_traits, bwd_args), gfx_arch));

        res.bwd_valid  = bwd_plan.is_valid() && bwd_plan.stages.size() >= 2;
        res.bwd_stages = static_cast<int>(bwd_plan.stages.size());

        std::cout << "  Fwd: " << std::fixed << std::setprecision(4) << res.fwd_ms << " ms, "
                  << std::setprecision(2) << res.fwd_tflops << " TFLOPS"
                  << " | Bwd plan: " << res.bwd_stages << " stages"
                  << (res.bwd_valid ? " (valid)" : " (invalid)") << "\n";

        results.push_back(res);
    }

    // Step 3: Summary table
    std::cout << "\nStep 3: Summary\n\n";
    std::cout << "  " << std::setw(7) << "Batch" << " | " << std::setw(7) << "SeqLen" << " | "
              << std::setw(10) << "Fwd(ms)" << " | " << std::setw(8) << "BwdPlan" << " | "
              << std::setw(10) << "FwdTFLOPS" << " | " << std::setw(6) << "Status" << "\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    bool all_passed = true;
    for(const auto& r : results)
    {
        std::cout << "  " << std::setw(7) << r.batch << " | " << std::setw(7) << r.seqlen << " | "
                  << std::fixed << std::setprecision(4) << std::setw(10) << r.fwd_ms << " | "
                  << std::setw(5) << r.bwd_stages << "stg" << " | " << std::setprecision(2)
                  << std::setw(10) << r.fwd_tflops << " | " << std::setw(6)
                  << (r.fwd_passed ? "PASS" : "FAIL") << "\n";
        if(!r.fwd_passed)
            all_passed = false;
    }

    print_separator();
    std::cout << "Status: " << (all_passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return all_passed ? 0 : 1;
}
