// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(batch_prefill_fmha_kernels,
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
                              .selection_rank(0),
                          "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 06: FMHA Batch Prefill",
                            "Declarative FMHA batch-prefill planning");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "1", "Batch size");
    args.add_option("--nhead", "16", "Number of heads");
    args.add_option("--seqlen", "128", "Query sequence length");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
    {
        return 0;
    }

    utils::print_header("Example 06: FMHA Batch Prefill");

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

    fmha_batch_prefill_traits traits{};
    traits.hdim_q           = hdim;
    traits.hdim_v           = hdim;
    traits.data_type        = "fp16";
    traits.is_group_mode    = true;
    traits.is_v_rowmajor    = true;
    traits.mask_type        = mask_enum::no_mask;
    traits.bias_type        = bias_enum::no_bias;
    traits.has_lse          = true;
    traits.kv_memory_layout = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    traits.kv_lookup_table  = ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    traits.page_size        = 16;

    fmha_batch_prefill_args fmha_args{};
    fmha_args.batch            = batch;
    fmha_args.seqlen_q         = seqlen;
    fmha_args.seqlen_k         = 1024;
    fmha_args.max_seqlen_q     = seqlen;
    fmha_args.hdim_q           = hdim;
    fmha_args.hdim_v           = hdim;
    fmha_args.nhead_q          = nhead;
    fmha_args.nhead_k          = nhead;
    fmha_args.num_total_pages  = 64;
    fmha_args.page_block_size  = 16;
    fmha_args.kv_memory_layout = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    fmha_args.kv_lookup_table = ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    fmha_args.kv_indptr       = reinterpret_cast<void*>(0x1);
    fmha_args.kv_page_indices = reinterpret_cast<void*>(0x1);
    fmha_args.kv_last_page_lens = reinterpret_cast<void*>(0x1);
    fmha_args.seqstart_q_ptr    = reinterpret_cast<void*>(0x1);

    auto plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_args), gfx_arch));

    if(!plan.is_valid() || plan.stages.size() != 1)
    {
        std::cerr << "Expected a single-stage batch-prefill plan\n";
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
