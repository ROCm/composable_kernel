// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 03: Benchmark
 *
 * Comprehensive performance benchmarking with warmup and statistics.
 *
 * Complexity: ★★★☆☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

int main(int argc, char** argv)
{
    print_header("Example 03: Benchmark");

    int M          = argc > 1 ? std::stoi(argv[1]) : 2048;
    int N          = argc > 2 ? std::stoi(argv[2]) : 2048;
    int K          = argc > 3 ? std::stoi(argv[3]) : 2048;
    int warmup     = 5;
    int iterations = 20;

    std::cout << "Problem: " << format_size(M, N, K) << "\n";
    std::cout << "Warmup: " << warmup << ", Iterations: " << iterations << "\n\n";

    // Setup kernel
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
    Problem problem(M, N, K);

    // Allocate
    GpuBuffer<ADataType> a_dev(M * K);
    GpuBuffer<BDataType> b_dev(K * N);
    GpuBuffer<CDataType> c_dev(M * N);

    std::vector<ADataType> a_host(M * K);
    std::vector<BDataType> b_host(K * N);
    fill_random(a_host.data(), M * K);
    fill_random(b_host.data(), K * N);

    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());

    // Warmup
    std::cout << "Warming up...\n";
    for(int i = 0; i < warmup; ++i)
    {
        c_dev.zero();
        (void)dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
    }

    // Benchmark
    std::cout << "Benchmarking...\n\n";
    std::vector<float> times;

    for(int i = 0; i < iterations; ++i)
    {
        c_dev.zero();
        times.push_back(dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr));
    }

    // Statistics
    std::sort(times.begin(), times.end());
    float min_t    = times.front();
    float max_t    = times.back();
    float median_t = times[iterations / 2];
    float avg_t    = 0;
    for(float t : times)
        avg_t += t;
    avg_t /= iterations;

    double flops = 2.0 * M * N * K;

    std::cout << "Results:\n";
    print_separator('-', 50);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Min:    " << min_t << " ms (" << std::setprecision(2)
              << (flops / (min_t * 1e-3)) / 1e12 << " TFLOPS)\n";
    std::cout << "  Avg:    " << std::setprecision(4) << avg_t << " ms (" << std::setprecision(2)
              << (flops / (avg_t * 1e-3)) / 1e12 << " TFLOPS)\n";
    std::cout << "  Median: " << std::setprecision(4) << median_t << " ms\n";
    std::cout << "  Max:    " << std::setprecision(4) << max_t << " ms\n";

    print_separator();
    std::cout << "Benchmark complete!\n";
    print_separator();

    return 0;
}
