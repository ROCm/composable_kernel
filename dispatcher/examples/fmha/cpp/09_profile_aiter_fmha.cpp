// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(
    aiter_profile_fmha_kernels,
    .add(FmhaSignature().family("fwd").dtype("fp16").mode("batch").vlayout("r").hdim(128).profile(
             "aiter_batch"),
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
             .padding(true, true, true, true),
         "gfx950")
        .add(FmhaSignature()
                 .family("fwd")
                 .dtype("fp16")
                 .mode("group")
                 .vlayout("r")
                 .hdim(128)
                 .profile("aiter_group"),
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
                 .family("fwd_pagedkv")
                 .dtype("fp16")
                 .mode("batch")
                 .vlayout("r")
                 .hdim(128)
                 .paged_kv(true)
                 .profile("aiter_cpp")
                 .kv_cache("vectorized", "sglang", 16),
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
                 .pipeline("qr_pagedkv")
                 .padding(true, true, true, true),
             "gfx950")
        .add(FmhaSignature()
                 .family("batch_prefill")
                 .dtype("fp16")
                 .mode("group")
                 .vlayout("r")
                 .hdim(128)
                 .paged_kv(true)
                 .profile("aiter_cpp")
                 .kv_cache("vectorized", "sglang", 16),
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
             "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 09: AITER-Profile FMHA",
                            "Declarative FMHA AITER-profile planning");
    args.add_option("--arch", "gfx950", "GPU architecture");
    if(!args.parse(argc, argv))
    {
        return 0;
    }

    const std::string gfx_arch = args.get("--arch", "gfx950");

    FmhaRegistry registry;
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    FmhaDispatcher dispatcher(&registry);

    std::cout << "AITER-profile FMHA kernels: " << registry.size() << "\n";

    fmha_fwd_traits batch_traits{};
    batch_traits.hdim_q        = 128;
    batch_traits.hdim_v        = 128;
    batch_traits.data_type     = "fp16";
    batch_traits.is_group_mode = false;
    batch_traits.is_v_rowmajor = true;
    batch_traits.mask_type     = mask_enum::no_mask;
    batch_traits.bias_type     = bias_enum::no_bias;
    batch_traits.qscale_type   = quant_scale_enum::no_scale;

    fmha_fwd_args batch_args{};
    batch_args.batch        = 1;
    batch_args.seqlen_q     = 128;
    batch_args.seqlen_k     = 128;
    batch_args.max_seqlen_q = 128;
    batch_args.hdim_q       = 128;
    batch_args.hdim_v       = 128;
    batch_args.nhead_q      = 16;
    batch_args.nhead_k      = 16;

    auto batch_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(batch_traits, batch_args), gfx_arch));

    fmha_batch_prefill_traits prefill_traits{};
    prefill_traits.hdim_q        = 128;
    prefill_traits.hdim_v        = 128;
    prefill_traits.data_type     = "fp16";
    prefill_traits.is_group_mode = true;
    prefill_traits.is_v_rowmajor = true;
    prefill_traits.mask_type     = mask_enum::no_mask;
    prefill_traits.bias_type     = bias_enum::no_bias;
    prefill_traits.kv_memory_layout =
        ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    prefill_traits.kv_lookup_table =
        ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    prefill_traits.page_size = 16;

    fmha_batch_prefill_args prefill_args{};
    prefill_args.batch           = 1;
    prefill_args.seqlen_q        = 128;
    prefill_args.seqlen_k        = 1024;
    prefill_args.max_seqlen_q    = 128;
    prefill_args.hdim_q          = 128;
    prefill_args.hdim_v          = 128;
    prefill_args.nhead_q         = 16;
    prefill_args.nhead_k         = 16;
    prefill_args.num_total_pages = 64;
    prefill_args.page_block_size = 16;
    prefill_args.kv_memory_layout =
        ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    prefill_args.kv_lookup_table =
        ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    prefill_args.kv_indptr         = reinterpret_cast<void*>(0x1);
    prefill_args.kv_page_indices   = reinterpret_cast<void*>(0x1);
    prefill_args.kv_last_page_lens = reinterpret_cast<void*>(0x1);
    prefill_args.seqstart_q_ptr    = reinterpret_cast<void*>(0x1);

    auto prefill_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(prefill_traits, prefill_args), gfx_arch));

    std::cout << "AITER batch stages: " << batch_plan.stages.size() << "\n";
    std::cout << "AITER prefill stages: " << prefill_plan.stages.size() << "\n";
    return (batch_plan.is_valid() && prefill_plan.is_valid()) ? 0 : 1;
}
