// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 15: Multi-Shape FMHA Sweep
//
// Demonstrates running a single FMHA kernel across multiple (batch, seqlen)
// combinations, producing a performance table. This pattern is useful for
// characterizing kernel behavior across the parameter space.
//
// Usage:
//   ./15_multi_shape_fmha
//   ./15_multi_shape_fmha --arch gfx942

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

DECL_FMHA_KERNEL_SET(multi_shape_fmha_kernels,
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

struct ShapeConfig
{
    int batch;
    int seqlen;
};

const ShapeConfig SHAPES[] = {
    {1, 64},
    {1, 128},
    {1, 256},
    {1, 512},
    {2, 64},
    {2, 128},
    {2, 256},
    {4, 64},
    {4, 128},
};

} // namespace

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 15: Multi-Shape FMHA",
                     "Sweep (batch, seqlen) combos with a single kernel");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--nhead", "8", "Number of heads");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int nhead            = args.get_int("--nhead", 8);
    const int hdim             = args.get_int("--hdim", 128);

    print_header("Example 15: Multi-Shape FMHA Sweep");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("multi_shape_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);
    dispatcher.set_benchmarking(true);
    dispatcher.set_timing(1, 3);

    // Step 2: Sweep shapes
    std::cout << "\nStep 2: Shape Sweep (nhead=" << nhead << ", hdim=" << hdim << ")\n\n";

    std::cout << "  " << std::setw(6) << "Batch" << " | " << std::setw(8) << "SeqLen" << " | "
              << std::setw(12) << "Elements" << " | " << std::setw(10) << "Time(ms)" << " | "
              << std::setw(10) << "TFLOPS" << " | " << std::setw(8) << "Status" << "\n";
    std::cout << "  " << std::string(66, '-') << "\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    int pass_count       = 0;
    int total            = 0;
    const int num_shapes = sizeof(SHAPES) / sizeof(SHAPES[0]);

    for(int si = 0; si < num_shapes; ++si)
    {
        const auto& shape = SHAPES[si];
        ++total;

        const float scale   = 1.0f / std::sqrt(static_cast<float>(hdim));
        const int64_t elems = static_cast<int64_t>(shape.batch) * nhead * shape.seqlen * hdim;

        GpuBuffer<FmhaDataType> q_dev(elems);
        GpuBuffer<FmhaDataType> k_dev(elems);
        GpuBuffer<FmhaDataType> v_dev(elems);
        GpuBuffer<FmhaDataType> o_dev(elems);

        std::vector<FmhaDataType> h_buf(elems);
        for(auto& x : h_buf)
            x = FmhaDataType(dist(rng));
        q_dev.copy_from_host(h_buf.data());
        for(auto& x : h_buf)
            x = FmhaDataType(dist(rng));
        k_dev.copy_from_host(h_buf.data());
        for(auto& x : h_buf)
            x = FmhaDataType(dist(rng));
        v_dev.copy_from_host(h_buf.data());
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

        fmha_args.seqlen_q        = shape.seqlen;
        fmha_args.seqlen_k        = shape.seqlen;
        fmha_args.batch           = shape.batch;
        fmha_args.max_seqlen_q    = shape.seqlen;
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

        fmha_args.nhead_stride_q         = shape.seqlen * hdim;
        fmha_args.nhead_stride_k         = shape.seqlen * hdim;
        fmha_args.nhead_stride_v         = shape.seqlen * hdim;
        fmha_args.nhead_stride_bias      = 0;
        fmha_args.nhead_stride_randval   = 0;
        fmha_args.nhead_stride_lse       = 0;
        fmha_args.nhead_stride_o         = shape.seqlen * hdim;
        fmha_args.nhead_stride_q_descale = 0;
        fmha_args.nhead_stride_k_descale = 0;
        fmha_args.nhead_stride_v_descale = 0;

        fmha_args.batch_stride_q         = nhead * shape.seqlen * hdim;
        fmha_args.batch_stride_k         = nhead * shape.seqlen * hdim;
        fmha_args.batch_stride_v         = nhead * shape.seqlen * hdim;
        fmha_args.batch_stride_bias      = 0;
        fmha_args.batch_stride_randval   = 0;
        fmha_args.batch_stride_lse       = 0;
        fmha_args.batch_stride_o         = nhead * shape.seqlen * hdim;
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

        bool ok       = false;
        float time_ms = 0.0f;
        try
        {
            time_ms = dispatcher.run_fwd(traits, fmha_args, nullptr);

            std::vector<FmhaDataType> o_host(elems);
            o_dev.copy_to_host(o_host.data());
            int nonzero = 0;
            for(int64_t i = 0; i < elems; ++i)
            {
                if(static_cast<float>(o_host[i]) != 0.0f)
                    ++nonzero;
            }
            ok = (nonzero > 0);
        }
        catch(const std::exception& e)
        {
            std::cerr << "  ERROR for B=" << shape.batch << " S=" << shape.seqlen << ": "
                      << e.what() << "\n";
        }

        auto problem =
            FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_args), gfx_arch);
        double tflops = static_cast<double>(problem.num_ops()) / (time_ms * 1e-3) / 1e12;

        std::cout << std::fixed;
        std::cout << "  " << std::setw(6) << shape.batch << " | " << std::setw(8) << shape.seqlen
                  << " | " << std::setw(12) << elems << " | " << std::setprecision(4)
                  << std::setw(10) << time_ms << " | " << std::setprecision(2) << std::setw(10)
                  << tflops << " | " << std::setw(8) << (ok ? "PASS" : "FAIL") << "\n";

        if(ok)
            ++pass_count;
    }

    // Summary
    print_separator();
    std::cout << "Results: " << pass_count << "/" << total << " shapes passed\n";
    std::cout << "Status: " << (pass_count == total ? "PASS" : "FAIL") << "\n";
    print_separator();

    return (pass_count == total) ? 0 : 1;
}
