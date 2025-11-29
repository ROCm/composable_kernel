// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 02: Multi-Size GEMM
 *
 * Demonstrates running GEMM with different problem sizes using a kernel set
 * optimized for various workloads.
 *
 * Build:
 *   python3 scripts/compile_gemm_examples.py examples/cpp/02_multi_size.cpp
 *
 * Complexity: ★★☆☆☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL SET: Multiple tile sizes for different problem sizes
// =============================================================================

DECL_KERNEL_SET(multi_size,
                .add("fp16", "rcr", 64, 64, 32)       // Small problems
                    .add("fp16", "rcr", 128, 128, 32) // Medium problems
                    .add("fp16", "rcr", 256, 256, 64) // Large problems
                    .add("fp16", "rcr", 128, 256, 32) // Rectangular (M < N)
                    .add("fp16", "rcr", 256, 128, 32) // Rectangular (M > N)
);

// =============================================================================
// MAIN
// =============================================================================

int main()
{
    print_header("Example 02: Multi-Size GEMM");

    // =========================================================================
    // Setup Registry and Dispatcher
    // =========================================================================
    std::cout << "\nSetup:\n";
    Registry registry;
    registry.set_name("multi_size_registry");

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
    Dispatcher dispatcher(&registry);
    std::cout << "  Registry: " << registry.size() << " kernel(s)\n";

    // =========================================================================
    // Run Multiple Problem Sizes
    // =========================================================================
    std::cout << "\nRunning multiple sizes:\n";
    print_separator();
    std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "K"
              << std::setw(12) << "Time(ms)" << std::setw(12) << "TFLOPS" << "\n";
    print_separator();

    // Test different sizes
    std::vector<std::tuple<int, int, int>> sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {1024, 2048, 512}, // Rectangular
        {2048, 1024, 512}, // Rectangular
    };

    bool all_passed = true;

    for(const auto& [M, N, K] : sizes)
    {
        Problem problem(M, N, K);

        // Allocate
        GpuBuffer<ADataType> a_dev(M * K);
        GpuBuffer<BDataType> b_dev(K * N);
        GpuBuffer<CDataType> c_dev(M * N);

        // Initialize
        std::vector<ADataType> a_host(M * K, ADataType(1.0f));
        std::vector<BDataType> b_host(K * N, BDataType(1.0f));
        a_dev.copy_from_host(a_host.data());
        b_dev.copy_from_host(b_host.data());
        c_dev.zero();

        // Run
        float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
        double tflops = calculate_tflops(M, N, K, time_ms);

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << K << std::setw(12)
                  << std::fixed << std::setprecision(4) << time_ms << std::setw(12)
                  << std::setprecision(2) << tflops << "\n";

        // Verify
        std::vector<CDataType> c_host(M * N);
        c_dev.copy_to_host(c_host.data());
        float expected = static_cast<float>(K);
        if(std::abs(static_cast<float>(c_host[0]) - expected) > 1.0f)
        {
            all_passed = false;
        }
    }

    print_separator();
    std::cout << "Status: " << (all_passed ? "ALL PASSED" : "SOME FAILED") << "\n";
    print_separator();

    return all_passed ? 0 : 1;
}
