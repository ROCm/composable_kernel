// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 24: Per-Receipt Registries
//
// Demonstrates:
//   1. Four DECL_FMHA_KERNEL_SET declarations, each named after a receipt
//   2. Each registered into a separate FmhaRegistry
//   3. Per-registry: kernel count, kernel names, plan a problem, selected kernel
//   4. GPU execution from the ck_default receipt (the basic working kernel)
//   5. Comparison table showing which features each receipt supports
//
// Receipt = a curated kernel set shipped to a specific downstream consumer.

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

// Receipt 1: CK default -- basic fp16, no mask, no bias
DECL_FMHA_KERNEL_SET(ck_default_kernels,
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

// Receipt 2: Flash forward -- fp16 with alibi bias
DECL_FMHA_KERNEL_SET(flash_fwd_kernels,
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
                              .qscale("no")
                              .profile("flash_fwd"),
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

// Receipt 3: PyTorch -- fp16 with elementwise bias
DECL_FMHA_KERNEL_SET(pytorch_kernels,
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
                              .qscale("no")
                              .profile("pytorch"),
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

// Receipt 4: AITER batch -- fp16 batch mode with LSE
DECL_FMHA_KERNEL_SET(aiter_batch_kernels,
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
                              .qscale("no")
                              .profile("aiter_batch"),
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

struct ReceiptInfo
{
    std::string name;
    std::string bias_desc;
    bool has_lse;
    FmhaRegistry registry;
};

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
                    scores[sk] /= sum_exp;

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
    ExampleArgs args("Example 24: Per-Receipt Registries",
                     "Curated kernel sets per downstream consumer");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--nhead", "4", "Number of heads");
    args.add_option("--seqlen", "64", "Sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 2);
    const int nhead            = args.get_int("--nhead", 4);
    const int seqlen           = args.get_int("--seqlen", 64);
    const int hdim             = args.get_int("--hdim", 128);
    const float scale          = 1.0f / std::sqrt(static_cast<float>(hdim));

    print_header("Example 24: Per-Receipt Registries");

    // Step 1: Create per-receipt registries
    std::cout << "\nStep 1: Create Per-Receipt Registries\n";
    std::cout << "  Global kernel sets: " << FmhaKernelSetRegistry::instance().size() << "\n";

    std::vector<ReceiptInfo> receipts;

    receipts.push_back({"ck_default", "none", false, FmhaRegistry()});
    receipts.back().registry.set_name("ck_default");
    REGISTER_GENERATED_KERNELS(receipts.back().registry, gfx_arch);

    receipts.push_back({"flash_fwd", "alibi", false, FmhaRegistry()});
    receipts.back().registry.set_name("flash_fwd");
    REGISTER_GENERATED_KERNELS(receipts.back().registry, gfx_arch);

    receipts.push_back({"pytorch", "elementwise", false, FmhaRegistry()});
    receipts.back().registry.set_name("pytorch");
    REGISTER_GENERATED_KERNELS(receipts.back().registry, gfx_arch);

    receipts.push_back({"aiter_batch", "none", true, FmhaRegistry()});
    receipts.back().registry.set_name("aiter_batch");
    REGISTER_GENERATED_KERNELS(receipts.back().registry, gfx_arch);

    // Step 2: Per-registry introspection
    std::cout << "\nStep 2: Per-Receipt Introspection\n";
    for(auto& r : receipts)
    {
        std::cout << "\n  Receipt: " << r.name << "\n";
        std::cout << "    Kernel count: " << r.registry.size() << "\n";

        auto all = r.registry.get_all();
        for(const auto& k : all)
        {
            std::cout << "    Kernel: " << k->get_name() << "\n";
        }
    }

    // Step 3: Plan a matching problem for each receipt
    std::cout << "\nStep 3: Plan per Receipt\n";

    struct PlanTest
    {
        std::string receipt_name;
        bias_enum bias;
        bool lse;
    };
    std::vector<PlanTest> plan_tests = {
        {"ck_default", bias_enum::no_bias, false},
        {"flash_fwd", bias_enum::alibi, false},
        {"pytorch", bias_enum::elementwise_bias, false},
        {"aiter_batch", bias_enum::no_bias, true},
    };

    for(std::size_t i = 0; i < plan_tests.size(); ++i)
    {
        const auto& pt = plan_tests[i];

        fmha_fwd_traits traits{};
        traits.hdim_q              = hdim;
        traits.hdim_v              = hdim;
        traits.data_type           = "fp16";
        traits.is_group_mode       = false;
        traits.is_v_rowmajor       = true;
        traits.has_logits_soft_cap = false;
        traits.mask_type           = mask_enum::no_mask;
        traits.bias_type           = pt.bias;
        traits.has_lse             = pt.lse;
        traits.has_dropout         = false;
        traits.qscale_type         = quant_scale_enum::no_scale;

        fmha_fwd_args fmha_args{};
        fmha_args.batch        = batch;
        fmha_args.seqlen_q     = seqlen;
        fmha_args.seqlen_k     = seqlen;
        fmha_args.max_seqlen_q = seqlen;
        fmha_args.hdim_q       = hdim;
        fmha_args.hdim_v       = hdim;
        fmha_args.nhead_q      = nhead;
        fmha_args.nhead_k      = nhead;

        FmhaDispatcher disp(&receipts[i].registry);
        auto plan = disp.plan(
            FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_args), gfx_arch));

        std::cout << "  " << pt.receipt_name << ": "
                  << (plan.is_valid() ? plan.stages[0].kernel_id : "NO MATCH") << "\n";
    }

    // Step 4: Comparison table
    std::cout << "\nStep 4: Receipt Feature Comparison\n\n";
    std::cout << "  " << std::setw(14) << "Receipt" << " | " << std::setw(14) << "Bias" << " | "
              << std::setw(5) << "LSE" << " | " << std::setw(8) << "Kernels" << "\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    struct CompRow
    {
        std::string name;
        std::string bias;
        std::string lse;
        std::size_t count;
    };
    std::vector<CompRow> comp = {
        {"ck_default", "none", "no", receipts[0].registry.size()},
        {"flash_fwd", "alibi", "no", receipts[1].registry.size()},
        {"pytorch", "elementwise", "no", receipts[2].registry.size()},
        {"aiter_batch", "none", "yes", receipts[3].registry.size()},
    };

    for(const auto& c : comp)
    {
        std::cout << "  " << std::setw(14) << c.name << " | " << std::setw(14) << c.bias << " | "
                  << std::setw(5) << c.lse << " | " << std::setw(8) << c.count << "\n";
    }

    // Step 5: GPU execution from ck_default
    std::cout << "\nStep 5: GPU Execution (ck_default receipt)\n";

    const int64_t q_elems = static_cast<int64_t>(batch) * nhead * seqlen * hdim;

    GpuBuffer<FmhaDataType> q_dev(q_elems);
    GpuBuffer<FmhaDataType> k_dev(q_elems);
    GpuBuffer<FmhaDataType> v_dev(q_elems);
    GpuBuffer<FmhaDataType> o_dev(q_elems);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<FmhaDataType> q_host(q_elems), k_host(q_elems), v_host(q_elems);
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

    fmha_fwd_traits run_traits{};
    run_traits.hdim_q              = hdim;
    run_traits.hdim_v              = hdim;
    run_traits.data_type           = "fp16";
    run_traits.is_group_mode       = false;
    run_traits.is_v_rowmajor       = true;
    run_traits.has_logits_soft_cap = false;
    run_traits.mask_type           = mask_enum::no_mask;
    run_traits.bias_type           = bias_enum::no_bias;
    run_traits.has_lse             = false;
    run_traits.has_dropout         = false;
    run_traits.qscale_type         = quant_scale_enum::no_scale;

    fmha_fwd_args run_args{};
    run_args.q_ptr = q_dev.get();
    run_args.k_ptr = k_dev.get();
    run_args.v_ptr = v_dev.get();
    run_args.o_ptr = o_dev.get();

    run_args.bias_ptr                   = nullptr;
    run_args.q_descale_ptr              = nullptr;
    run_args.k_descale_ptr              = nullptr;
    run_args.v_descale_ptr              = nullptr;
    run_args.rand_val_ptr               = nullptr;
    run_args.lse_ptr                    = nullptr;
    run_args.sink_ptr                   = nullptr;
    run_args.block_scale_seqstart_q_ptr = nullptr;
    run_args.block_scale_seqstart_k_ptr = nullptr;

    run_args.seqlen_q        = seqlen;
    run_args.seqlen_k        = seqlen;
    run_args.batch           = batch;
    run_args.max_seqlen_q    = seqlen;
    run_args.hdim_q          = hdim;
    run_args.hdim_v          = hdim;
    run_args.nhead_q         = nhead;
    run_args.nhead_k         = nhead;
    run_args.scale_s         = scale;
    run_args.logits_soft_cap = 0.0f;

    run_args.stride_q       = hdim;
    run_args.stride_k       = hdim;
    run_args.stride_v       = hdim;
    run_args.stride_bias    = 0;
    run_args.stride_randval = 0;
    run_args.stride_o       = hdim;

    run_args.nhead_stride_q         = seqlen * hdim;
    run_args.nhead_stride_k         = seqlen * hdim;
    run_args.nhead_stride_v         = seqlen * hdim;
    run_args.nhead_stride_bias      = 0;
    run_args.nhead_stride_randval   = 0;
    run_args.nhead_stride_lse       = 0;
    run_args.nhead_stride_o         = seqlen * hdim;
    run_args.nhead_stride_q_descale = 0;
    run_args.nhead_stride_k_descale = 0;
    run_args.nhead_stride_v_descale = 0;

    run_args.batch_stride_q         = nhead * seqlen * hdim;
    run_args.batch_stride_k         = nhead * seqlen * hdim;
    run_args.batch_stride_v         = nhead * seqlen * hdim;
    run_args.batch_stride_bias      = 0;
    run_args.batch_stride_randval   = 0;
    run_args.batch_stride_lse       = 0;
    run_args.batch_stride_o         = nhead * seqlen * hdim;
    run_args.batch_stride_q_descale = 0;
    run_args.batch_stride_k_descale = 0;
    run_args.batch_stride_v_descale = 0;

    run_args.window_size_left    = -1;
    run_args.window_size_right   = -1;
    run_args.sink_size           = 0;
    run_args.mask_type           = 0;
    run_args.min_seqlen_q        = 0;
    run_args.p_drop              = 0.0f;
    run_args.s_randval           = false;
    run_args.drop_seed_offset    = std::make_pair(uint64_t(0), uint64_t(0));
    run_args.block_scale_size_q  = 0;
    run_args.block_scale_size_kv = 0;

    FmhaDispatcher ck_disp(&receipts[0].registry);
    ck_disp.set_benchmarking(true);
    ck_disp.set_timing(1, 3);

    bool passed = false;
    try
    {
        float time_ms = ck_disp.run_fwd(run_traits, run_args, nullptr);
        std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";

        std::vector<FmhaDataType> o_host(q_elems);
        o_dev.copy_to_host(o_host.data());

        std::vector<float> q_f32(q_elems), k_f32(q_elems), v_f32(q_elems), o_ref(q_elems, 0.0f);
        for(int64_t i = 0; i < q_elems; ++i)
            q_f32[i] = static_cast<float>(q_host[i]);
        for(int64_t i = 0; i < q_elems; ++i)
            k_f32[i] = static_cast<float>(k_host[i]);
        for(int64_t i = 0; i < q_elems; ++i)
            v_f32[i] = static_cast<float>(v_host[i]);

        cpu_attention_fwd(
            q_f32, k_f32, v_f32, o_ref, batch, nhead, seqlen, seqlen, hdim, hdim, scale);

        double max_abs_err = 0.0;
        int errors         = 0;
        const double rtol  = 1e-2;
        const double atol  = 1e-2;
        for(int64_t i = 0; i < q_elems; ++i)
        {
            float gpu_val  = static_cast<float>(o_host[i]);
            float ref_val  = o_ref[i];
            double abs_err = std::abs(gpu_val - ref_val);
            max_abs_err    = std::max(max_abs_err, abs_err);
            if(abs_err > atol + rtol * std::abs(ref_val))
                ++errors;
        }
        std::cout << "  Max abs error: " << std::scientific << max_abs_err << "\n";
        std::cout << "  Errors: " << errors << " / " << q_elems << "\n";
        passed = (errors == 0);
    }
    catch(const std::exception& e)
    {
        std::cerr << "  ERROR: " << e.what() << "\n";
    }

    print_separator();
    std::cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return passed ? 0 : 1;
}
