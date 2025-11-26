// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 01: Basic GEMM
 *
 * The simplest example - runs a single GEMM operation via dispatcher.
 *
 * Complexity: ★☆☆☆☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

int main()
{
    print_header("Example 01: Basic GEMM");

    // Step 1: Setup kernel from force-included header
    std::cout << "Step 1: Setup kernel...\n";
    std::cout << "  Kernel: " << KERNEL_NAME << "\n";
    std::cout << "  Tile: " << SelectedKernel::TileM << "x" << SelectedKernel::TileN << "x"
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

    // Step 2: Run GEMM
    std::cout << "Step 2: Run GEMM 1024x1024x1024...\n";

    const int M = 1024, N = 1024, K = 1024;
    Problem problem(M, N, K);

    GpuBuffer<ADataType> a_dev(M * K);
    GpuBuffer<BDataType> b_dev(K * N);
    GpuBuffer<CDataType> c_dev(M * N);

    std::vector<ADataType> a_host(M * K, ADataType(1.0f));
    std::vector<BDataType> b_host(K * N, BDataType(1.0f));

    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());
    c_dev.zero();

    Dispatcher dispatcher;
    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);

    double tflops = calculate_tflops(M, N, K, time_ms);
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n\n";

    // Step 3: Verify
    std::cout << "Step 3: Verify...\n";
    std::vector<CDataType> c_host(M * N);
    c_dev.copy_to_host(c_host.data());

    float expected = static_cast<float>(K);
    float actual   = static_cast<float>(c_host[0]);
    bool passed    = std::abs(actual - expected) < 1.0f;

    std::cout << "  C[0,0] = " << actual << " (expected " << expected << ")\n";
    std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n\n";

    print_separator();
    std::cout << "Example 01 complete!\n";
    print_separator();

    return passed ? 0 : 1;
}
