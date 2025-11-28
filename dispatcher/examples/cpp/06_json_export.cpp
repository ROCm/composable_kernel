// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 06: JSON Export
 *
 * Demonstrates exporting registry information to JSON format.
 *
 * Build:
 *   python3 scripts/build_with_kernels.py examples/cpp/06_json_export.cpp
 *
 * Complexity: ★★☆☆☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL SET
// =============================================================================

DECL_KERNEL_SET(json_export,
                .add("fp16", "rcr", 64, 64, 32)
                    .add("fp16", "rcr", 128, 128, 32)
                    .add("fp16", "rcr", 256, 256, 64));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    print_header("Example 06: JSON Export");

    std::string output_file = "registry.json";
    if(argc > 1)
    {
        output_file = argv[1];
    }

    // =========================================================================
    // Setup Registry
    // =========================================================================
    std::cout << "\nSetting up registry...\n";
    Registry registry;
    registry.set_name("json_export_registry");

    KernelConfig config =
        KernelConfig::fp16_rcr()
            .tile(SelectedKernel::TileM, SelectedKernel::TileN, SelectedKernel::TileK)
            .wave(SelectedKernel::WarpPerBlock_M,
                  SelectedKernel::WarpPerBlock_N,
                  SelectedKernel::WarpPerBlock_K)
            .warp_tile(
                SelectedKernel::WarpTileM, SelectedKernel::WarpTileN, SelectedKernel::WarpTileK);

    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            config.build_key(), KERNEL_NAME);

    registry.register_kernel(kernel);

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
    // Also export kernel set declarations
    // =========================================================================
    std::cout << "\nKernel Set Declarations:\n";
    print_separator();
    const auto& kernel_set = KernelSetRegistry::instance().get("json_export");
    for(const auto& decl : kernel_set.declarations())
    {
        std::cout << "  " << decl.name() << "\n";
    }
    print_separator();

    return 0;
}
