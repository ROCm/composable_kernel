// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 05: Custom Heuristics
 *
 * Demonstrates custom kernel selection heuristics for different workloads.
 *
 * Build:
 *   python3 scripts/build_with_kernels.py examples/cpp/05_heuristics.cpp
 *
 * Complexity: ★★★☆☆
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
// KERNEL SET: Variety of tile sizes for heuristic selection
// =============================================================================

DECL_KERNEL_SET(heuristics,
                .add("fp16", "rcr", 64, 64, 32)       // Small tile - low latency
                    .add("fp16", "rcr", 128, 128, 32) // Medium tile - balanced
                    .add("fp16", "rcr", 256, 256, 64) // Large tile - high throughput
);

// =============================================================================
// Custom Heuristic Functions
// =============================================================================

// Heuristic: Prefer small tiles for small problems, large tiles for large
float size_based_heuristic(const Problem& problem, const KernelInstancePtr& kernel)
{
    int64_t total_elements = problem.M * problem.N;
    const auto& key        = kernel->get_key();
    int tile_m             = key.algorithm.tile_shape[0];
    int tile_n             = key.algorithm.tile_shape[1];
    int tile_size          = tile_m * tile_n;

    // Score based on how well tile size matches problem size
    float ideal_tile = std::sqrt(static_cast<float>(total_elements) / 64.0f);
    float tile_score = 1.0f / (1.0f + std::abs(tile_size - ideal_tile) / ideal_tile);

    return tile_score;
}

// Heuristic: Prefer tiles that evenly divide the problem
float divisibility_heuristic(const Problem& problem, const KernelInstancePtr& kernel)
{
    const auto& key = kernel->get_key();
    int tile_m      = key.algorithm.tile_shape[0];
    int tile_n      = key.algorithm.tile_shape[1];

    bool divides_m = (problem.M % tile_m) == 0;
    bool divides_n = (problem.N % tile_n) == 0;

    return (divides_m && divides_n) ? 1.0f : 0.5f;
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
    dispatcher.set_strategy(SelectionStrategy::Heuristic);
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
            std::cout << "  Selected tile: " << key.algorithm.tile_shape[0] << "x"
                      << key.algorithm.tile_shape[1] << "\n";
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
                  << "\n\n";
    }

    print_separator();
    std::cout << "Heuristic functions available:\n";
    std::cout << "  - size_based_heuristic: Matches tile to problem size\n";
    std::cout << "  - divisibility_heuristic: Prefers evenly-dividing tiles\n";
    print_separator();

    return 0;
}
