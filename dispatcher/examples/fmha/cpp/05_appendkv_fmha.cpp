// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(appendkv_fmha_kernels,
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
                              // Stage 0 (Q*K^T): m0=seqlen_q, n0=seqlen_k, k0=hdim_q
                              .tile_m0(64)
                              .tile_n0(64)
                              .tile_k0(128)
                              // Stage 1 (Attn*V): n1=hdim_v, k1=seqlen_k, k0max=alignment
                              .tile_n1(128)
                              .tile_k1(0)
                              .tile_k0max(0)
                              .pipeline("appendkv")
                              .padding(true, true, true, true)
                              .selection_rank(0),
                          "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 05: FMHA AppendKV", "Declarative FMHA append-KV planning");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--batch", "1", "Batch size");
    args.add_option("--nhead", "16", "Number of heads");
    args.add_option("--seqlen", "1", "Sequence length (tokens to append)");
    args.add_option("--hdim", "128", "Head dimension");
    if(!args.parse(argc, argv))
    {
        return 0;
    }

    utils::print_header("Example 05: FMHA AppendKV");

    const std::string gfx_arch = args.get("--arch", "gfx950");
    const int batch            = args.get_int("--batch", 1);
    const int nhead            = args.get_int("--nhead", 16);
    const int seqlen           = args.get_int("--seqlen", 1);
    const int hdim             = args.get_int("--hdim", 128);

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaRegistry registry;
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);

    // Step 2: Plan
    std::cout << "\nStep 2: Plan\n";

    fmha_fwd_appendkv_traits traits{};
    traits.hdim_q        = hdim;
    traits.hdim_v        = hdim;
    traits.data_type     = "fp16";
    traits.is_v_rowmajor = true;
    traits.rope_type     = rope_enum::interleaved;

    fmha_fwd_appendkv_args fmha_args{};
    fmha_args.seqlen_q        = seqlen;
    fmha_args.seqlen_knew     = seqlen;
    fmha_args.batch           = batch;
    fmha_args.hdim_q          = hdim;
    fmha_args.hdim_v          = hdim;
    fmha_args.nhead_q         = nhead;
    fmha_args.nhead_k         = nhead;
    fmha_args.rotary_dim      = hdim;
    fmha_args.rotary_cos_ptr  = reinterpret_cast<void*>(0x1);
    fmha_args.rotary_sin_ptr  = reinterpret_cast<void*>(0x1);
    fmha_args.block_table_ptr = reinterpret_cast<void*>(0x1);
    fmha_args.page_block_size = 16;

    auto plan = dispatcher.plan(
        FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_args), gfx_arch));

    if(!plan.is_valid() || plan.stages.size() != 1)
    {
        std::cerr << "Expected a single-stage append-KV plan\n";
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
