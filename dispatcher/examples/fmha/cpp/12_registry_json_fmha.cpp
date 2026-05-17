// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <fstream>
#include <iostream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(
    registry_json_fmha_kernels,
    .add(FmhaSignature().family("fwd").dtype("fp16").mode("batch").vlayout("r").hdim(128),
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
                 .family("fwd_pagedkv")
                 .dtype("fp16")
                 .mode("batch")
                 .vlayout("r")
                 .hdim(128)
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
                 .pipeline("qr_pagedkv")
                 .padding(true, true, true, true),
             "gfx950")
        .add(FmhaSignature().family("bwd_dq_dk_dv").dtype("fp16").mode("batch").hdim(128),
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
             "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 12: Registry JSON FMHA",
                            "Declarative FMHA registry JSON export");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--output", "", "Write JSON to file (optional)");
    if(!args.parse(argc, argv))
    {
        return 0;
    }

    utils::print_header("Example 12: Registry JSON FMHA");

    const std::string gfx_arch    = args.get("--arch", "gfx950");
    const std::string output_path = args.get("--output", "");

    // Step 1: Register kernels
    std::cout << "\nStep 1: Register Kernels\n";
    FmhaRegistry registry;
    registry.set_name("registry_json_fmha");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    // Step 2: Export JSON
    std::cout << "\nStep 2: Export JSON\n";
    std::string json = registry.export_json(true);
    std::cout << "  JSON size: " << json.size() << " bytes\n";
    std::cout << json.substr(0, std::min<std::size_t>(json.size(), 240)) << "\n";

    // Step 3: Write to file (if --output specified)
    if(!output_path.empty())
    {
        std::cout << "\nStep 3: Write to File\n";
        std::ofstream ofs(output_path);
        if(!ofs.is_open())
        {
            std::cerr << "  ERROR: Cannot open " << output_path << " for writing\n";
            return 1;
        }
        ofs << json;
        ofs.close();
        std::cout << "  Written to: " << output_path << "\n";
        std::cout << "  File size:  " << json.size() << " bytes\n";
    }

    utils::print_separator();
    return registry.size() > 0 ? 0 : 1;
}
