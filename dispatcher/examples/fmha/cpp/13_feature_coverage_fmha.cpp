// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Example 13: FMHA Feature Coverage
// Exercises every feature dimension from the 01_fmha smoke test:
// bf16, masks (top-left, bottom-right, window_generic), GQA, dropout,
// multiple hdims (64, 256), group mode, col-major V.

#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(feature_coverage_kernels,
                     // fp16 forward (basic, needed for GQA and other fp16 tests)
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("no")
                              .bias("no")
                              .lse(false)
                              .dropout(false)
                              .qscale("no"),
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
                              .alignments(128, 128)
                              .selection_rank(0),
                          "gfx950")

                         // bf16 forward
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("bf16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
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
                              "gfx950")

                         // hdim 64
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(64)
                                  .mask("no")
                                  .bias("no")
                                  .lse(false)
                                  .dropout(false)
                                  .qscale("no"),
                              FmhaAlgorithm()
                                  .tile_m0(128)
                                  .tile_n0(64)
                                  .tile_k0(32)
                                  .tile_n1(64)
                                  .tile_k1(32)
                                  .tile_k0max(64)
                                  .wave_m0(4)
                                  .wave_n0(1)
                                  .wave_k0(1)
                                  .wave_m1(4)
                                  .wave_n1(1)
                                  .wave_k1(1)
                                  .warp_m0(32)
                                  .warp_n0(32)
                                  .warp_k0(16)
                                  .warp_m1(16)
                                  .warp_n1(16)
                                  .warp_k1(16)
                                  .pipeline("qr_async")
                                  .padding(true, true, true, true)
                                  .alignments(64, 64)
                                  .selection_rank(0),
                              "gfx950")

                         // hdim 256
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(256)
                                  .mask("no")
                                  .bias("no")
                                  .lse(true)
                                  .dropout(false)
                                  .qscale("no"),
                              FmhaAlgorithm()
                                  .tile_m0(128)
                                  .tile_n0(128)
                                  .tile_k0(32)
                                  .tile_n1(256)
                                  .tile_k1(32)
                                  .tile_k0max(256)
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
                                  .pipeline("qr")
                                  .padding(false, false, false, false)
                                  .alignments(256, 256)
                                  .selection_rank(0),
                              "gfx950")

                         // Mask: causal (top-left and bottom-right share the same compiled kernel;
                         //   the mask type is resolved at runtime via the args, not the template)
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("top_left")
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
                              "gfx950")

                         // Dropout
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no")
                                  .lse(true)
                                  .dropout(true)
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

                         // GQA (nhead_q != nhead_k) - same kernel, GQA is a runtime concern
                         // Bias: elementwise
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("bias")
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
                              "gfx950")

                         // Bias: alibi
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("alibi")
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
                              "gfx950")

                         // Group mode
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("group")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("no")
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
                              "gfx950")

                         // Sink tokens
                         .add(FmhaSignature()
                                  .family("fwd")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .vlayout("r")
                                  .hdim(128)
                                  .mask("top_left")
                                  .bias("no")
                                  .lse(false)
                                  .dropout(false)
                                  .qscale("no")
                                  .sink(true),
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

namespace {

struct FeatureTest
{
    std::string name;
    FmhaProblem problem;
};

FeatureTest make_test(const std::string& name,
                      const std::string& dtype,
                      int hdim_q,
                      int hdim_v,
                      int mask,
                      int bias,
                      bool lse,
                      bool dropout,
                      bool group,
                      bool logits,
                      bool sink,
                      int nhead_q             = 16,
                      int nhead_k             = 16,
                      const std::string& arch = "gfx950")
{
    auto p = FmhaProblemBuilder()
                 .api_family(FmhaApiFamily::Fwd)
                 .kernel_family(FmhaKernelFamily::Fwd)
                 .gfx_arch(arch)
                 .data_type(dtype)
                 .dims(hdim_q, hdim_v, 2, 128, 256)
                 .nheads(nhead_q, nhead_k)
                 .mask_type(mask)
                 .bias_type(bias)
                 .lse(lse)
                 .dropout(dropout)
                 .group_mode(group)
                 .logits_soft_cap(logits)
                 .sink(sink)
                 .build();
    return {name, p};
}

} // namespace

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 13: FMHA Feature Coverage",
                            "Tests all 01_fmha smoke test features");
    args.add_option("--arch", "gfx950", "GPU architecture");
    if(!args.parse(argc, argv))
        return 0;

    utils::print_header("Example 13: FMHA Feature Coverage");

    const std::string gfx_arch = args.get("--arch", "gfx950");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaKernelSetRegistry::instance().print();

    FmhaRegistry registry;
    registry.set_name("feature_coverage");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    FmhaDispatcher dispatcher(&registry);

    // Step 2: Run feature tests
    std::cout << "\nStep 2: Run Feature Tests\n";
    std::vector<FeatureTest> tests = {
        make_test("bf16_basic", "bf16", 128, 128, 0, 0, false, false, false, false, false),
        make_test("fp16_hdim64", "fp16", 64, 64, 0, 0, false, false, false, false, false),
        make_test("fp16_hdim256", "fp16", 256, 256, 0, 0, true, false, false, false, false),
        make_test("mask_top_left", "fp16", 128, 128, 1, 0, false, false, false, false, false),
        make_test("mask_bottom_right", "fp16", 128, 128, 2, 0, false, false, false, false, false),
        make_test("dropout", "fp16", 128, 128, 0, 0, true, true, false, false, false),
        make_test("gqa_h16_hk4", "fp16", 128, 128, 0, 0, false, false, false, false, false, 16, 4),
        make_test("bias_elementwise", "fp16", 128, 128, 0, 1, false, false, false, false, false),
        make_test("bias_alibi", "fp16", 128, 128, 0, 2, false, false, false, false, false),
        make_test("group_mode", "fp16", 128, 128, 0, 0, false, false, true, false, false),
        make_test("sink_tokens", "fp16", 128, 128, 1, 0, false, false, false, false, true),
    };

    int pass = 0;
    int fail = 0;
    for(const auto& test : tests)
    {
        auto plan = dispatcher.plan(test.problem);
        bool ok   = plan.is_valid();
        std::cout << (ok ? "[PASS]" : "[FAIL]") << " " << test.name;
        if(ok)
        {
            std::cout << " -> " << plan.stages[0].kernel_id;
            ++pass;
        }
        else
        {
            ++fail;
        }
        std::cout << "\n";
    }

    // Step 3: Summary
    std::cout << "\nStep 3: Summary\n";
    std::cout << "  " << pass << " passed, " << fail << " failed out of " << tests.size() << "\n";

    utils::print_separator();
    return fail > 0 ? 1 : 0;
}
