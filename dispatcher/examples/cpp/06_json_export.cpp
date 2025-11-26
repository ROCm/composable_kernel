// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 06: JSON Export
 *
 * Export kernel registry to JSON for debugging and analysis.
 *
 * Complexity: ★★☆☆☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

namespace kernel_config {
using ADataType   = ck_tile::fp16_t;
using BDataType   = ck_tile::fp16_t;
using CDataType   = ck_tile::fp16_t;
using AccDataType = float;
} // namespace kernel_config

int main(int argc, char** argv)
{
    print_header("Example 06: JSON Export");

    using namespace kernel_config;

    std::string output_file = argc > 1 ? argv[1] : "kernels.json";

    // Register kernel
    std::cout << "Step 1: Registering kernel...\n";

    KernelKeyBuilder builder = KernelKeyBuilder::fp16_rcr();
    builder.tile_m           = SelectedKernel::TileM;
    builder.tile_n           = SelectedKernel::TileN;
    builder.tile_k           = SelectedKernel::TileK;
    builder.wave_m           = SelectedKernel::WarpPerBlock_M;
    builder.wave_n           = SelectedKernel::WarpPerBlock_N;
    builder.wave_k           = SelectedKernel::WarpPerBlock_K;
    builder.warp_m           = SelectedKernel::WarpTileM;
    builder.warp_n           = SelectedKernel::WarpTileN;
    builder.warp_k           = SelectedKernel::WarpTileK;
    builder.block_size       = SelectedKernel::BlockSize;

    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            builder.build(), KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Registry::Priority::High);
    std::cout << "  Registered: " << KERNEL_NAME << "\n\n";

    // Export
    std::cout << "Step 2: Exporting to JSON...\n";
    std::string json = Registry::instance().export_json(true);

    std::ofstream file(output_file);
    if(file.is_open())
    {
        file << json;
        file.close();
        std::cout << "  Saved to: " << output_file << "\n\n";
    }

    // Preview
    std::cout << "Step 3: Preview:\n";
    print_separator('-', 60);
    std::cout << json.substr(0, 500);
    if(json.length() > 500)
        std::cout << "\n...";
    std::cout << "\n";
    print_separator('-', 60);

    print_separator();
    std::cout << "JSON export complete!\n";
    print_separator();

    return 0;
}
