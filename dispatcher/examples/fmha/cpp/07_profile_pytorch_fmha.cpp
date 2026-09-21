// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(pytorch_profile_fmha_kernels,
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("no")
                              .bias("bias")
                              .profile("pytorch"),
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
                              .warp_k0(32)
                              .warp_m1(16)
                              .warp_n1(16)
                              .warp_k1(16)
                              .pipeline("qr")
                              .padding(true, true, true, true),
                          "gfx950")
                         .add(FmhaSignature()
                                  .family("fwd_splitkv")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
                                  .lse(true)
                                  .profile("pytorch"),
                              FmhaAlgorithm()
                                  .tile_m0(64)
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
                                  .warp_m0(16)
                                  .warp_n0(16)
                                  .warp_k0(16)
                                  .warp_m1(16)
                                  .warp_n1(16)
                                  .warp_k1(16)
                                  .pipeline("qr")
                                  .padding(true, true, true, true)
                                  .max_splits_log2(6),
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
                                  .profile("pytorch"),
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
                                  .max_splits_log2(6),
                              "gfx950")
                         .add(FmhaSignature()
                                  .family("fwd_appendkv")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .profile("pytorch"),
                              FmhaAlgorithm()
                                  .tile_m0(64)
                                  .tile_n0(64)
                                  .tile_k0(128)
                                  .tile_n1(128)
                                  .tile_k1(0)
                                  .tile_k0max(0)
                                  .padding(false, true, true, false)
                                  .pipeline("appendkv"),
                              "gfx950")
                         .add(FmhaSignature()
                                  .family("bwd_dot_do_o")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
                                  .profile("pytorch"),
                              FmhaAlgorithm()
                                  .tile_m0(64)
                                  .tile_n0(128)
                                  .tile_k0(32)
                                  .tile_n1(0)
                                  .tile_k1(0)
                                  .tile_k0max(0)
                                  .padding(true, true, true, true),
                              "gfx950")
                         .add(FmhaSignature()
                                  .family("bwd_dq_dk_dv")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
                                  .profile("pytorch"),
                              FmhaAlgorithm()
                                  .tile_m0(16)
                                  .tile_n0(128)
                                  .tile_k0(128)
                                  .tile_n1(16)
                                  .tile_k1(128)
                                  .tile_k0max(32)
                                  .wave(1, 4, 1, 4, 1, 1, 1, 4, 1)
                                  .warp(16, 16, 32, 16, 16, 16, 16, 16, 16)
                                  .padding(true, true, true, true),
                              "gfx950")
                         .add(FmhaSignature()
                                  .family("bwd_convert_dq")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
                                  .profile("pytorch"),
                              FmhaAlgorithm()
                                  .tile_m0(64)
                                  .tile_n0(128)
                                  .tile_k0(0)
                                  .tile_n1(0)
                                  .tile_k1(0)
                                  .tile_k0max(0)
                                  .padding(true, true, true, true),
                              "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 07: PyTorch-Profile FMHA",
                            "Declarative FMHA PyTorch-profile planning");
    args.add_option("--arch", "gfx950", "GPU architecture");
    if(!args.parse(argc, argv))
    {
        return 0;
    }

    const std::string gfx_arch = args.get("--arch", "gfx950");

    FmhaRegistry registry;
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    FmhaDispatcher dispatcher(&registry);

    std::cout << "PyTorch-profile FMHA kernels: " << registry.size() << "\n";

    fmha_fwd_traits fwd_traits{};
    fwd_traits.hdim_q        = 128;
    fwd_traits.hdim_v        = 128;
    fwd_traits.data_type     = "fp16";
    fwd_traits.is_group_mode = false;
    fwd_traits.is_v_rowmajor = true;
    fwd_traits.mask_type     = mask_enum::no_mask;
    fwd_traits.bias_type     = bias_enum::elementwise_bias;
    fwd_traits.qscale_type   = quant_scale_enum::no_scale;

    fmha_fwd_args fwd_args{};
    fwd_args.batch        = 1;
    fwd_args.seqlen_q     = 128;
    fwd_args.seqlen_k     = 128;
    fwd_args.max_seqlen_q = 128;
    fwd_args.hdim_q       = 128;
    fwd_args.hdim_v       = 128;
    fwd_args.nhead_q      = 16;
    fwd_args.nhead_k      = 16;

    auto fwd_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(fwd_traits, fwd_args), gfx_arch));

    fmha_bwd_traits bwd_traits{};
    bwd_traits.hdim_q        = 128;
    bwd_traits.hdim_v        = 128;
    bwd_traits.data_type     = "fp16";
    bwd_traits.is_group_mode = false;
    bwd_traits.mask_type     = mask_enum::no_mask;
    bwd_traits.bias_type     = bias_enum::no_bias;

    fmha_bwd_args bwd_args{};
    bwd_args.batch        = 1;
    bwd_args.seqlen_q     = 128;
    bwd_args.seqlen_k     = 128;
    bwd_args.max_seqlen_q = 128;
    bwd_args.max_seqlen_k = 128;
    bwd_args.hdim_q       = 128;
    bwd_args.hdim_v       = 128;
    bwd_args.nhead_q      = 16;
    bwd_args.nhead_k      = 16;

    auto bwd_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(bwd_traits, bwd_args), gfx_arch));

    std::cout << "Forward plan stages: " << fwd_plan.stages.size() << "\n";
    std::cout << "Backward plan stages: " << bwd_plan.stages.size() << "\n";
    return (fwd_plan.is_valid() && bwd_plan.is_valid()) ? 0 : 1;
}
