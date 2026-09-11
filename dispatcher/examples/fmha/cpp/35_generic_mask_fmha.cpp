// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 35: FMHA Forward with Generic/Window Mask
//
// Demonstrates forward kernel with generic (window) mask. Uses
// window_size_left and window_size_right: for each query i, attend only to
// keys in [i - left, i + right]. -1 means unbounded. Planning only.

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;

DECL_FMHA_KERNEL_SET(generic_mask_fmha_kernels,
                     // Forward with generic/window mask
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("generic") // window mask via left/right
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

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 35: FMHA Generic Mask", "Window mask via left/right params");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "2", "Batch size");
    args.add_option("--nhead", "4", "Number of heads");
    args.add_option("--seqlen", "128", "Sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    args.add_option("--window_left", "64", "Window size left (-1=unbounded)");
    args.add_option("--window_right", "0", "Window size right (-1=unbounded)");
    if(!args.parse(argc, argv))
        return 0;

    const std::string gfx_arch  = args.get("--arch", "gfx950");
    const int batch             = args.get_int("--batch", 2);
    const int nhead             = args.get_int("--nhead", 4);
    const int seqlen            = args.get_int("--seqlen", 128);
    const int hdim              = args.get_int("--hdim", 128);
    const int window_size_left  = args.get_int("--window_left", 64);
    const int window_size_right = args.get_int("--window_right", 0);

    print_header("Example 35: FMHA Generic Mask");

    std::cout << "\nStep 1: Register Kernels\n";
    FmhaRegistry registry;
    registry.set_name("generic_mask_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    std::cout << "\nStep 2: Plan\n";
    FmhaDispatcher dispatcher(&registry);
    fmha_fwd_traits traits{};
    traits.hdim_q        = hdim;
    traits.hdim_v        = hdim;
    traits.data_type     = "fp16";
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::window_generic;
    traits.bias_type     = bias_enum::no_bias;
    traits.has_lse       = false;
    traits.has_dropout   = false;
    traits.qscale_type   = quant_scale_enum::no_scale;

    fmha_fwd_args fwd_args{};
    fwd_args.batch             = batch;
    fwd_args.seqlen_q          = seqlen;
    fwd_args.seqlen_k          = seqlen;
    fwd_args.nhead_q           = nhead;
    fwd_args.nhead_k           = nhead;
    fwd_args.hdim_q            = hdim;
    fwd_args.hdim_v            = hdim;
    fwd_args.window_size_left  = window_size_left;
    fwd_args.window_size_right = window_size_right;

    auto plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(traits, fwd_args), gfx_arch));
    std::cout << "  Plan valid: " << (plan.is_valid() ? "yes" : "no") << "\n";

    std::cout << "\nStep 3: Window Mask Params\n";
    std::cout << "  window_size_left=" << window_size_left
              << ", window_size_right=" << window_size_right << "\n";
    std::cout << "  Query i attends to keys in [i-left, i+right]. -1 = unbounded.\n";

    print_separator();
    return 0;
}
