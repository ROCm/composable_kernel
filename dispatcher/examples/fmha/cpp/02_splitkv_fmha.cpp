// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(splitkv_fmha_kernels,
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
                              .pipeline("qr")
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

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 02: FMHA Split-KV", "Declarative FMHA split-KV planning");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "1", "Batch size");
    args.add_option("--nhead", "16", "Number of heads");
    args.add_option("--seqlen", "128", "Query sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
    {
        return 0;
    }

    utils::print_header("Example 02: FMHA Split-KV");

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 1);
    const int nhead            = args.get_int("--nhead", 16);
    const int seqlen           = args.get_int("--seqlen", 128);
    const int hdim             = args.get_int("--hdim", 128);

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaRegistry registry;
    registry.set_name("splitkv_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);

    // Step 2: Plan
    std::cout << "\nStep 2: Plan\n";

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

    fmha_fwd_splitkv_args fmha_args{};
    fmha_args.seqlen_q     = seqlen;
    fmha_args.seqlen_k     = 2048;
    fmha_args.batch        = batch;
    fmha_args.max_seqlen_q = seqlen;
    fmha_args.hdim_q       = hdim;
    fmha_args.hdim_v       = hdim;
    fmha_args.nhead_q      = nhead;
    fmha_args.nhead_k      = nhead;
    fmha_args.num_splits   = 8;

    auto problem = FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_args), gfx_arch);
    auto plan    = dispatcher.plan(problem);

    if(!plan.is_valid() || plan.stages.size() != 2)
    {
        std::cerr << "Expected a two-stage split-KV plan\n";
        return 1;
    }

    // Step 3: Results
    std::cout << "\nStep 3: Results\n";
    for(const auto& stage : plan.stages)
    {
        std::cout << "  " << to_string(stage.family) << " -> " << stage.kernel_id << "\n";
    }

    utils::print_separator();
    return 0;
}
