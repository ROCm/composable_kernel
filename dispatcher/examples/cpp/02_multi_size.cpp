// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 02: Multi-Size Testing
 *
 * Tests multiple problem sizes to understand performance scaling.
 *
 * Complexity: ★★☆☆☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

int main()
{
    print_header("Example 02: Multi-Size Testing");

    // Setup kernel
    std::cout << "Kernel: " << KERNEL_NAME << "\n";
    std::cout << "Tile: " << SelectedKernel::TileM << "x" << SelectedKernel::TileN << "x"
              << SelectedKernel::TileK << "\n\n";

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
    Registry::instance().register_kernel(kernel);

    Dispatcher dispatcher;

    // Test sizes
    std::vector<std::tuple<int, int, int>> sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
    };

    std::cout << std::setw(20) << "Size" << " | " << std::setw(12) << "Time (ms)" << " | "
              << std::setw(10) << "TFLOPS" << "\n";
    print_separator('-', 50);

    for(const auto& [M, N, K] : sizes)
    {
        Problem problem(M, N, K);

        GpuBuffer<ADataType> a_dev(M * K);
        GpuBuffer<BDataType> b_dev(K * N);
        GpuBuffer<CDataType> c_dev(M * N);

        std::vector<ADataType> a_host(M * K, ADataType(1.0f));
        std::vector<BDataType> b_host(K * N, BDataType(1.0f));

        a_dev.copy_from_host(a_host.data());
        b_dev.copy_from_host(b_host.data());
        c_dev.zero();

        float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
        double tflops = calculate_tflops(M, N, K, time_ms);

        std::cout << std::setw(20) << format_size(M, N, K) << " | " << std::setw(12) << std::fixed
                  << std::setprecision(4) << time_ms << " | " << std::setw(10)
                  << std::setprecision(2) << tflops << "\n";
    }

    print_separator();
    std::cout << "Multi-size testing complete!\n";
    print_separator();

    return 0;
}
