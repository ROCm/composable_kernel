// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(kvcache_fmha_kernels,
                     .add(FmhaSignature()
                              .family("fwd_pagedkv")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("no")
                              .bias("no")
                              .lse(false)
                              .dropout(false)
                              .qscale("no")
                              .paged_kv(true)
                              .kv_cache("vectorized", "sglang", 16),
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
                              .pipeline("qr_pagedkv")
                              .padding(true, true, true, true)
                              .selection_rank(0),
                          "gfx950")
                         .add(FmhaSignature()
                                  .family("fwd_appendkv")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .rope("inter")
                                  .paged_kv(true)
                                  .kv_cache("vectorized", "sglang", 16),
                              FmhaAlgorithm()
                                  .tile_m0(64)
                                  .tile_n0(64)
                                  .tile_k0(128)
                                  .tile_n1(128)
                                  .tile_k1(0)
                                  .tile_k0max(0)
                                  .pipeline("appendkv")
                                  .padding(true, true, true, true)
                                  .selection_rank(0),
                              "gfx950")
                         .add(FmhaSignature()
                                  .family("batch_prefill")
                                  .dtype("fp16")
                                  .mode("group")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
                                  .lse(true)
                                  .dropout(false)
                                  .qscale("no")
                                  .paged_kv(true)
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
                                  .padding(true, true, true, true)
                                  .selection_rank(0),
                              "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 03: FMHA KV-Cache", "Declarative FMHA KV-cache planning");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "1", "Batch size");
    args.add_option("--nhead", "16", "Number of heads");
    args.add_option("--seqlen", "128", "Prefill query sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
    {
        return 0;
    }

    utils::print_header("Example 03: FMHA KV-Cache");

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

    // Step 2: Plan PagedKV (decode)
    std::cout << "\nStep 2: Plan PagedKV (decode)\n";

    fmha_fwd_pagedkv_traits paged_traits{};
    paged_traits.hdim_q        = hdim;
    paged_traits.hdim_v        = hdim;
    paged_traits.data_type     = "fp16";
    paged_traits.is_group_mode = false;
    paged_traits.is_v_rowmajor = true;
    paged_traits.mask_type     = mask_enum::no_mask;
    paged_traits.bias_type     = bias_enum::no_bias;
    paged_traits.use_pagedkv   = true;

    fmha_fwd_pagedkv_args paged_args{};
    paged_args.seqlen_q        = 1;
    paged_args.seqlen_k        = 1024;
    paged_args.batch           = batch;
    paged_args.max_seqlen_q    = 1;
    paged_args.hdim_q          = hdim;
    paged_args.hdim_v          = hdim;
    paged_args.nhead_q         = nhead;
    paged_args.nhead_k         = nhead;
    paged_args.block_table_ptr = reinterpret_cast<void*>(0x1);
    paged_args.page_block_size = 16;

    auto paged_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(paged_traits, paged_args), gfx_arch));

    // Step 3: Plan AppendKV
    std::cout << "\nStep 3: Plan AppendKV\n";

    fmha_fwd_appendkv_traits append_traits{};
    append_traits.hdim_q        = hdim;
    append_traits.hdim_v        = hdim;
    append_traits.data_type     = "fp16";
    append_traits.is_v_rowmajor = true;
    append_traits.rope_type     = rope_enum::interleaved;

    fmha_fwd_appendkv_args append_args{};
    append_args.seqlen_q        = 1;
    append_args.seqlen_knew     = 1;
    append_args.batch           = batch;
    append_args.hdim_q          = hdim;
    append_args.hdim_v          = hdim;
    append_args.nhead_q         = nhead;
    append_args.nhead_k         = nhead;
    append_args.rotary_dim      = hdim;
    append_args.rotary_cos_ptr  = reinterpret_cast<void*>(0x1);
    append_args.rotary_sin_ptr  = reinterpret_cast<void*>(0x1);
    append_args.block_table_ptr = reinterpret_cast<void*>(0x1);
    append_args.page_block_size = 16;

    auto append_plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(append_traits, append_args), gfx_arch));

    // Step 4: Plan BatchPrefill
    std::cout << "\nStep 4: Plan BatchPrefill\n";

    fmha_batch_prefill_traits prefill_traits{};
    prefill_traits.hdim_q        = hdim;
    prefill_traits.hdim_v        = hdim;
    prefill_traits.data_type     = "fp16";
    prefill_traits.is_group_mode = true;
    prefill_traits.is_v_rowmajor = true;
    prefill_traits.mask_type     = mask_enum::no_mask;
    prefill_traits.bias_type     = bias_enum::no_bias;
    prefill_traits.has_lse       = true;
    prefill_traits.kv_memory_layout =
        ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    prefill_traits.kv_lookup_table =
        ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    prefill_traits.page_size = 16;

    fmha_batch_prefill_args prefill_args{};
    prefill_args.batch           = batch;
    prefill_args.seqlen_q        = seqlen;
    prefill_args.seqlen_k        = 1024;
    prefill_args.max_seqlen_q    = seqlen;
    prefill_args.hdim_q          = hdim;
    prefill_args.hdim_v          = hdim;
    prefill_args.nhead_q         = nhead;
    prefill_args.nhead_k         = nhead;
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

    // Step 5: Results
    std::cout << "\nStep 5: Results\n";
    std::cout << "  PagedKV stages:    " << paged_plan.stages.size() << "\n";
    std::cout << "  AppendKV stages:   " << append_plan.stages.size() << "\n";
    std::cout << "  BatchPrefill stages: " << prefill_plan.stages.size() << "\n";

    utils::print_separator();
    return (paged_plan.is_valid() && append_plan.is_valid() && prefill_plan.is_valid()) ? 0 : 1;
}
