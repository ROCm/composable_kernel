// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 34: FMHA Backward with GQA (Grouped Query Attention)
//
// Demonstrates backward with nhead_q=8, nhead_k=2 (4:1 ratio). GQA is a
// runtime property: each KV head is shared by multiple Q heads. Backward
// handles head indexing via nhead_stride_dk/dv so dK/dV are reduced across
// the Q-head group. Planning only.

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;

DECL_FMHA_KERNEL_SET(bwd_gqa_fmha_kernels,
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("top_left")
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
                         .add(FmhaSignature()
                                  .family("bwd_dot_do_o")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("top_left")
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
                         .add(FmhaSignature()
                                  .family("bwd_dq_dk_dv")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("top_left")
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
                                  .mask("top_left")
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
    ExampleArgs args("Example 34: FMHA Backward GQA", "nhead_q=8, nhead_k=2 (4:1 ratio)");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--nhead_q", "8", "Query heads");
    args.add_option("--nhead_k", "2", "KV heads (GQA ratio = nhead_q/nhead_k)");
    args.add_option("--seqlen", "128", "Sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 2);
    const int nhead_q          = args.get_int("--nhead_q", 8);
    const int nhead_k          = args.get_int("--nhead_k", 2);
    const int seqlen           = args.get_int("--seqlen", 128);
    const int hdim             = args.get_int("--hdim", 128);

    print_header("Example 34: FMHA Backward GQA");

    std::cout << "\nStep 1: Register Kernels\n";
    FmhaRegistry registry;
    registry.set_name("bwd_gqa_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    std::cout << "\nStep 2: Plan (GQA nhead_q=" << nhead_q << ", nhead_k=" << nhead_k << ")\n";
    FmhaDispatcher dispatcher(&registry);
    fmha_bwd_traits traits{};
    traits.hdim_q           = hdim;
    traits.hdim_v           = hdim;
    traits.data_type        = "fp16";
    traits.is_group_mode    = false;
    traits.mask_type        = mask_enum::mask_top_left;
    traits.bias_type        = bias_enum::no_bias;
    traits.has_dbias        = false;
    traits.has_dropout      = false;
    traits.is_store_randval = false;
    traits.is_deterministic = false;

    fmha_bwd_args bwd_args{};
    bwd_args.batch    = batch;
    bwd_args.seqlen_q = seqlen;
    bwd_args.seqlen_k = seqlen;
    bwd_args.hdim_q   = hdim;
    bwd_args.hdim_v   = hdim;
    bwd_args.nhead_q  = nhead_q;
    bwd_args.nhead_k  = nhead_k;

    auto plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(traits, bwd_args), gfx_arch));
    std::cout << "  Plan valid: " << (plan.is_valid() ? "yes" : "no") << "\n";

    std::cout << "\nStep 3: GQA Backward Head Indexing\n";
    std::cout << "  Q heads " << nhead_q << ", KV heads " << nhead_k
              << " -> each KV head shared by " << (nhead_q / nhead_k) << " Q heads.\n";
    std::cout << "  dK/dV reduced across Q-head group via nhead_stride.\n";

    print_separator();
    return 0;
}
