// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(receipt_alias_fmha_kernels,
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .bias("alibi")
                              .receipt(2),
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
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .bias("bias")
                                  .receipt(4),
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
                                  .warp_k0(32)
                                  .warp_m1(16)
                                  .warp_n1(16)
                                  .warp_k1(16)
                                  .pipeline("qr")
                                  .padding(true, true, true, true),
                              "gfx950")
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .receipt(100),
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
                                  .padding(true, true, true, true),
                              "gfx950")
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp32")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .receipt(800),
                              FmhaAlgorithm()
                                  .tile_m0(32)
                                  .tile_n0(128)
                                  .tile_k0(32)
                                  .tile_n1(128)
                                  .tile_k1(16)
                                  .tile_k0max(128)
                                  .wave_m0(2)
                                  .wave_n0(1)
                                  .wave_k0(1)
                                  .wave_m1(2)
                                  .wave_n1(1)
                                  .wave_k1(1)
                                  .warp_m0(16)
                                  .warp_n0(16)
                                  .warp_k0(16)
                                  .warp_m1(16)
                                  .warp_n1(16)
                                  .warp_k1(16)
                                  .pipeline("qr")
                                  .padding(true, true, true, true),
                              "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 11: Receipt Aliases FMHA",
                            "Declarative FMHA receipt-alias planning");
    args.add_option("--arch", "gfx950", "GPU architecture");
    if(!args.parse(argc, argv))
    {
        return 0;
    }

    const std::string gfx_arch = args.get("--arch", "gfx950");

    FmhaRegistry registry;
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    FmhaDispatcher dispatcher(&registry);

    std::cout << "Receipt-alias FMHA kernels: " << registry.size() << "\n";

    fmha_fwd_traits traits{};
    traits.hdim_q        = 128;
    traits.hdim_v        = 128;
    traits.data_type     = "fp16";
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::no_mask;
    traits.bias_type     = bias_enum::no_bias;
    traits.qscale_type   = quant_scale_enum::no_scale;

    fmha_fwd_args fmha_args{};
    fmha_args.batch        = 1;
    fmha_args.seqlen_q     = 128;
    fmha_args.seqlen_k     = 128;
    fmha_args.max_seqlen_q = 128;
    fmha_args.hdim_q       = 128;
    fmha_args.hdim_v       = 128;
    fmha_args.nhead_q      = 16;
    fmha_args.nhead_k      = 16;

    auto plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_args), gfx_arch));

    std::cout << "Receipt-alias plan stages: " << plan.stages.size() << "\n";
    return plan.is_valid() ? 0 : 1;
}
