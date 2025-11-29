// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 05: Custom Heuristics
 *
 * Demonstrates custom kernel selection heuristics for different workloads.
 *
 * Build:
 *   python3 scripts/compile_gemm_examples.py examples/cpp/05_heuristics.cpp
 *
 * Complexity: ★★★☆☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL SET: Variety of tile sizes for heuristic selection
// =============================================================================

DECL_KERNEL_SET(heuristics,
                .add("fp16", "rcr", 64, 64, 32)       // Small tile - low latency
                    .add("fp16", "rcr", 128, 128, 32) // Medium tile - balanced
                    .add("fp16", "rcr", 256, 256, 64) // Large tile - high throughput
);

// =============================================================================
// Custom Heuristic: Returns kernel names ranked by expected performance
// =============================================================================

// Heuristic: Size-based selection - returns kernels ranked for problem size
std::vector<std::string> size_based_heuristic(const Problem& problem)
{
    std::vector<std::string> ranked_kernels;
    int64_t total_elements = problem.M * problem.N;

    // Classify problem size and return appropriate kernels
    if(total_elements < 10000)
    {
        // Small problems: prefer small tiles for low latency
        ranked_kernels = {"gemm_64x64", "gemm_128x128", "gemm_256x256"};
    }
    else if(total_elements < 1000000)
    {
        // Medium problems: balanced approach
        ranked_kernels = {"gemm_128x128", "gemm_64x64", "gemm_256x256"};
    }
    else
    {
        // Large problems: prefer large tiles for throughput
        ranked_kernels = {"gemm_256x256", "gemm_128x128", "gemm_64x64"};
    }

    return ranked_kernels;
}

// =============================================================================
// MAIN
// =============================================================================

int main()
{
    print_header("Example 05: Custom Heuristics");

    // =========================================================================
    // Setup
    // =========================================================================
    Registry registry;
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

    // Create dispatcher with heuristic selection
    Dispatcher dispatcher(&registry);
    dispatcher.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
    dispatcher.set_heuristic(size_based_heuristic);

    std::cout << "\nSetup:\n";
    std::cout << "  Registry: " << registry.size() << " kernel(s)\n";
    std::cout << "  Strategy: Heuristic (size-based)\n";

    // =========================================================================
    // Test Different Problem Sizes
    // =========================================================================
    std::cout << "\nTesting heuristic selection:\n";
    print_separator();

    std::vector<std::tuple<int, int, int>> sizes = {
        {128, 128, 64},     // Small
        {512, 512, 256},    // Medium
        {2048, 2048, 1024}, // Large
    };

    for(const auto& [M, N, K] : sizes)
    {
        Problem problem(M, N, K);
        auto selected = dispatcher.select_kernel(problem);

        std::cout << "Problem " << M << "x" << N << "x" << K << ":\n";
        if(selected)
        {
            const auto& key = selected->get_key();
            std::cout << "  Selected tile: " << key.algorithm.tile_shape.m << "x"
                      << key.algorithm.tile_shape.n << "x" << key.algorithm.tile_shape.k << "\n";
        }

        // Actually run it
        GpuBuffer<ADataType> a_dev(M * K);
        GpuBuffer<BDataType> b_dev(K * N);
        GpuBuffer<CDataType> c_dev(M * N);

        std::vector<ADataType> a_host(M * K, ADataType(1.0f));
        std::vector<BDataType> b_host(K * N, BDataType(1.0f));
        a_dev.copy_from_host(a_host.data());
        b_dev.copy_from_host(b_host.data());
        c_dev.zero();

        float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
        std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
        std::cout << "  TFLOPS: " << std::setprecision(2) << calculate_tflops(M, N, K, time_ms)
                  << "\n";
        print_separator();
    }

    // =========================================================================
    // Demonstrate manual heuristic logic
    // =========================================================================
    std::cout << "\nHeuristic Decision Logic:\n";
    print_separator();

    std::cout << "Problem Size Classification:\n";
    std::cout << "  Small  (<10K elements):  Prefer 64x64   tiles for low latency\n";
    std::cout << "  Medium (<1M elements):   Prefer 128x128 tiles for balance\n";
    std::cout << "  Large  (>1M elements):   Prefer 256x256 tiles for throughput\n";

    print_separator();
    std::cout << "Heuristics enable adaptive kernel selection based on:\n";
    std::cout << "  - Problem size and shape\n";
    std::cout << "  - Hardware characteristics\n";
    std::cout << "  - Memory bandwidth requirements\n";
    std::cout << "  - Compute vs memory bound workloads\n";

    return 0;
}
