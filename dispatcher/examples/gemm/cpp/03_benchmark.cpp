// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 03: GEMM Benchmarking
 *
 * Runs GEMM multiple times to get accurate timing statistics.
 *
 * Build:
 *   python3 scripts/compile_gemm_examples.py examples/cpp/03_benchmark.cpp
 *
 * Complexity: ★★☆☆☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL SET: High-performance kernels for benchmarking
// =============================================================================

DECL_KERNEL_SET(benchmark, .add("fp16", "rcr", 128, 128, 32).add("fp16", "rcr", 256, 256, 64));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    print_header("Example 03: GEMM Benchmarking");

    // Parse args
    int M = 4096, N = 4096, K = 4096;
    int warmup = 5, iterations = 100;

    if(argc >= 4)
    {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if(argc >= 5)
        iterations = std::atoi(argv[4]);

    std::cout << "\nConfiguration:\n";
    std::cout << "  Problem:    " << M << " x " << N << " x " << K << "\n";
    std::cout << "  Warmup:     " << warmup << " iterations\n";
    std::cout << "  Benchmark:  " << iterations << " iterations\n";

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
    Dispatcher dispatcher(&registry);

    std::cout << "  Kernel:     " << kernel->get_name() << "\n";

    // Allocate
    GpuBuffer<ADataType> a_dev(M * K);
    GpuBuffer<BDataType> b_dev(K * N);
    GpuBuffer<CDataType> c_dev(M * N);

    std::vector<ADataType> a_host(M * K, ADataType(0.5f));
    std::vector<BDataType> b_host(K * N, BDataType(0.5f));
    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());

    Problem problem(M, N, K);

    // =========================================================================
    // Warmup
    // =========================================================================
    std::cout << "\nWarmup...\n";
    for(int i = 0; i < warmup; ++i)
    {
        c_dev.zero();
        dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
    }

    // =========================================================================
    // Benchmark
    // =========================================================================
    std::cout << "Benchmarking...\n";
    std::vector<float> times;
    times.reserve(iterations);

    for(int i = 0; i < iterations; ++i)
    {
        c_dev.zero();
        float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
        times.push_back(time_ms);
    }

    // =========================================================================
    // Statistics
    // =========================================================================
    std::sort(times.begin(), times.end());

    float min_time    = times.front();
    float max_time    = times.back();
    float median_time = times[times.size() / 2];
    float mean_time   = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();

    // Remove outliers for stable mean
    size_t trim = times.size() / 10; // 10% from each end
    float trimmed_mean =
        std::accumulate(times.begin() + trim, times.end() - trim, 0.0f) / (times.size() - 2 * trim);

    double flops         = 2.0 * M * N * K;
    double min_tflops    = (flops / (min_time / 1000.0)) / 1e12;
    double median_tflops = (flops / (median_time / 1000.0)) / 1e12;
    double mean_tflops   = (flops / (mean_time / 1000.0)) / 1e12;

    print_separator();
    std::cout << "Benchmark Results (" << iterations << " iterations):\n";
    print_separator();
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Min time:     " << min_time << " ms (" << std::setprecision(2) << min_tflops
              << " TFLOPS)\n";
    std::cout << std::setprecision(4);
    std::cout << "  Max time:     " << max_time << " ms\n";
    std::cout << "  Mean time:    " << mean_time << " ms (" << std::setprecision(2) << mean_tflops
              << " TFLOPS)\n";
    std::cout << std::setprecision(4);
    std::cout << "  Median time:  " << median_time << " ms (" << std::setprecision(2)
              << median_tflops << " TFLOPS)\n";
    std::cout << std::setprecision(4);
    std::cout << "  Trimmed mean: " << trimmed_mean << " ms\n";
    print_separator();

    return 0;
}
