// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(bwd_fmha_kernels,
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
                              // Stage 0 (Q*K^T): m0=seqlen_q, n0=seqlen_k, k0=hdim_q
                              .tile_m0(64)
                              .tile_n0(128)
                              .tile_k0(32)
                              // Stage 1 (Attn*V): n1=hdim_v, k1=seqlen_k, k0max=alignment
                              .tile_n1(0)
                              .tile_k1(0)
                              .tile_k0max(0)
                              .padding(true, true, true, true)
                              .selection_rank(0),
                          "gfx950")
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

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 04: FMHA Backward", "Declarative FMHA backward planning");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "1", "Batch size");
    args.add_option("--nhead", "16", "Number of heads");
    args.add_option("--seqlen", "128", "Sequence length (Q and K)");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
    {
        return 0;
    }

    utils::print_header("Example 04: FMHA Backward");

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 1);
    const int nhead            = args.get_int("--nhead", 16);
    const int seqlen           = args.get_int("--seqlen", 128);
    const int hdim             = args.get_int("--hdim", 128);

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaRegistry registry;
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);

    // Step 2: Plan
    std::cout << "\nStep 2: Plan\n";

    fmha_bwd_traits traits{};
    traits.hdim_q           = hdim;
    traits.hdim_v           = hdim;
    traits.data_type        = "fp16";
    traits.is_group_mode    = false;
    traits.mask_type        = mask_enum::no_mask;
    traits.bias_type        = bias_enum::no_bias;
    traits.has_dbias        = false;
    traits.has_dropout      = false;
    traits.is_store_randval = false;
    traits.is_deterministic = false;

    fmha_bwd_args bwd_args{};
    bwd_args.batch        = batch;
    bwd_args.seqlen_q     = seqlen;
    bwd_args.seqlen_k     = seqlen;
    bwd_args.max_seqlen_q = seqlen;
    bwd_args.max_seqlen_k = seqlen;
    bwd_args.hdim_q       = hdim;
    bwd_args.hdim_v       = hdim;
    bwd_args.nhead_q      = nhead;
    bwd_args.nhead_k      = nhead;

    auto plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(traits, bwd_args), gfx_arch));

    if(!plan.is_valid() || plan.stages.size() < 2)
    {
        std::cerr << "Expected a multi-stage backward plan\n";
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
