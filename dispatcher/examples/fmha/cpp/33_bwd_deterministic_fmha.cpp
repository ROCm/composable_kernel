// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 33: FMHA Backward Deterministic vs Non-Deterministic
//
// Demonstrates two backward kernel sets: one deterministic (bit-identical
// results across runs) and one non-deterministic (faster, atomic reductions).
// Planning only.

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;

DECL_FMHA_KERNEL_SET(bwd_deterministic_fmha_kernels,
                     // Forward: causal + LSE for backward
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
                         // Backward: deterministic (bit-identical across runs)
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
                                  .deterministic(true),
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
                                  .deterministic(true),
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
                                  .deterministic(true),
                              FmhaAlgorithm()
                                  .tile_m0(64)
                                  .tile_n0(128)
                                  .tile_k0(0)
                                  .tile_n1(0)
                                  .tile_k1(0)
                                  .tile_k0max(0)
                                  .padding(true, true, true, true)
                                  .selection_rank(0),
                              "gfx950")
                         // Backward: non-deterministic (faster, atomic reductions)
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
                                  .selection_rank(1),
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
                                  .selection_rank(1),
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
                                  .selection_rank(1),
                              "gfx950"));

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 33: FMHA Backward Deterministic",
                     "Deterministic vs non-deterministic backward");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--nhead", "4", "Number of heads");
    args.add_option("--seqlen", "128", "Sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 2);
    const int nhead            = args.get_int("--nhead", 4);
    const int seqlen           = args.get_int("--seqlen", 128);
    const int hdim             = args.get_int("--hdim", 128);

    print_header("Example 33: FMHA Backward Deterministic");

    std::cout << "\nStep 1: Register Kernels\n";
    FmhaRegistry registry;
    registry.set_name("bwd_deterministic_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    std::cout << "\nStep 2: Plan (deterministic)\n";
    FmhaDispatcher dispatcher(&registry);
    fmha_bwd_traits det_traits{};
    det_traits.hdim_q           = hdim;
    det_traits.hdim_v           = hdim;
    det_traits.data_type        = "fp16";
    det_traits.is_group_mode    = false;
    det_traits.mask_type        = mask_enum::mask_top_left;
    det_traits.bias_type        = bias_enum::no_bias;
    det_traits.has_dbias        = false;
    det_traits.has_dropout      = false;
    det_traits.is_store_randval = false;
    det_traits.is_deterministic = true;

    fmha_bwd_args bwd_args{};
    bwd_args.batch    = batch;
    bwd_args.seqlen_q = seqlen;
    bwd_args.seqlen_k = seqlen;
    bwd_args.hdim_q   = hdim;
    bwd_args.hdim_v   = hdim;
    bwd_args.nhead_q  = nhead;
    bwd_args.nhead_k  = nhead;

    auto det_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(det_traits, bwd_args), gfx_arch));
    std::cout << "  Deterministic plan valid: " << (det_plan.is_valid() ? "yes" : "no") << "\n";

    std::cout << "\nStep 3: Plan (non-deterministic)\n";
    det_traits.is_deterministic = false;
    auto nondet_plan            = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(det_traits, bwd_args), gfx_arch));
    std::cout << "  Non-deterministic plan valid: " << (nondet_plan.is_valid() ? "yes" : "no")
              << "\n";

    std::cout << "\nStep 4: Deterministic Mode\n";
    std::cout << "  deterministic=true: bit-identical across runs (reproducible).\n";
    std::cout << "  deterministic=false: faster, uses atomic reductions.\n";

    print_separator();
    return 0;
}
