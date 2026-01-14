// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 05: JSON Export
 *
 * Demonstrates exporting registry information to JSON format.
 *
 * Build: cd dispatcher/build && cmake .. && make gemm_05_json_export
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// KERNEL SET: Multiple kernels for JSON export demo
// =============================================================================

DECL_KERNEL_SET(json_export_kernels,
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(64, 64, 32)
                         .wave(2, 2, 1)
                         .warp(32, 32, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx942")
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 128, 64)
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942"));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 05: JSON Export", "Export registry information to JSON format");
    args.add_option("--output", "registry.json", "Output JSON file path");
    args.add_option("--arch", "gfx942", "GPU architecture");
    args.add_flag("--list", "List all kernel sets");

    if(!args.parse(argc, argv))
        return 0;

    print_header("Example 05: JSON Export");

    std::string gfx_arch = args.get("--arch", "gfx942");

    if(args.has("--list"))
    {
        std::cout << "\nDeclared Kernel Sets:\n";
        KernelSetRegistry::instance().print();
        return 0;
    }

    std::string output_file = args.get("--output", "registry.json");

    // =========================================================================
    // Setup Registry
    // =========================================================================
    std::cout << "\nSetting up registry...\n";
    Registry registry;
    registry.set_name("json_export_registry");

    REGISTER_GENERATED_KERNELS(registry, gfx_arch);

    std::cout << "  Registry: " << registry.get_name() << "\n";
    std::cout << "  Kernels:  " << registry.size() << "\n";

    // =========================================================================
    // Export to JSON
    // =========================================================================
    std::cout << "\nExporting to JSON...\n";

    std::string json = registry.export_json(true);

    std::cout << "\nJSON Preview (first 500 chars):\n";
    print_separator();
    std::cout << json.substr(0, std::min(size_t(500), json.size()));
    if(json.size() > 500)
        std::cout << "\n...";
    std::cout << "\n";
    print_separator();

    // Write to file
    std::ofstream file(output_file);
    if(file.is_open())
    {
        file << json;
        file.close();
        std::cout << "\nExported to: " << output_file << "\n";
        std::cout << "File size: " << json.size() << " bytes\n";
    }
    else
    {
        std::cerr << "Failed to write to: " << output_file << "\n";
        return 1;
    }

    // =========================================================================
    // Also show kernel set declarations
    // =========================================================================
    std::cout << "\nKernel Set Declarations:\n";
    print_separator();
    KernelSetRegistry::instance().print();
    print_separator();

    return 0;
}
