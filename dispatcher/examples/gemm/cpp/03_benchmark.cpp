// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 03: Advanced GEMM Benchmarking
 *
 * Demonstrates all available benchmark parameters matching CK Tile stream_config:
 *   - warmup: Number of warmup iterations (default: 5)
 *   - iterations: Number of benchmark iterations (default: 100)
 *
 * Build: cd dispatcher/build && cmake .. && make gemm_03_benchmark
 * Usage: ./gemm_03_benchmark [--size N] [--warmup N] [--iterations N]
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// KERNEL SET: High-performance kernels for benchmarking
// =============================================================================

DECL_KERNEL_SET(benchmark_kernels,
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(128, 128, 64)
                         .wave(2, 2, 1)
                         .warp(32, 32, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx942")
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(256, 128, 64)
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
    ExampleArgs args("Example 03: GEMM Benchmarking",
                     "Runs GEMM multiple times for accurate timing");
    args.add_option("--size", "4096", "Problem size MxNxK");
    args.add_option("--warmup", "5", "Warmup iterations");
    args.add_option("--iterations", "100", "Benchmark iterations");
    args.add_option("--arch", "gfx942", "GPU architecture");

    if(!args.parse(argc, argv))
    {
        return 0;
    }

    int M                = args.get_int("--size", 4096);
    int N                = M;
    int K                = M;
    int warmup           = args.get_int("--warmup", 5);
    int iterations       = args.get_int("--iterations", 100);
    std::string gfx_arch = args.get("--arch", "gfx942");

    print_header("Example 03: GEMM Benchmarking");

    std::cout << "\nConfiguration:\n";
    std::cout << "  Problem:    " << M << " x " << N << " x " << K << "\n";
    std::cout << "  Warmup:     " << warmup << " iterations\n";
    std::cout << "  Benchmark:  " << iterations << " iterations\n";

    // =========================================================================
    // Setup Registry and Dispatcher
    // =========================================================================
    Registry registry;
    generated::register_03_benchmark_kernels(registry, gfx_arch);
    Dispatcher dispatcher(&registry);

    std::cout << "  Kernels:    " << registry.size() << " registered\n";

    // Select kernel and print its name
    Problem problem(M, N, K);
    auto selected = dispatcher.select_kernel(problem);
    if(selected)
    {
        std::cout << "  Selected:   " << selected->get_name() << "\n";
    }

    // =========================================================================
    // Allocate and initialize
    // =========================================================================
    using DataType = ck_tile::fp16_t;

    GpuBuffer<DataType> a_dev(M * K);
    GpuBuffer<DataType> b_dev(K * N);
    GpuBuffer<DataType> c_dev(M * N);

    std::vector<DataType> a_host(M * K, DataType(0.5f));
    std::vector<DataType> b_host(K * N, DataType(0.5f));
    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());

    // =========================================================================
    // Warmup
    // =========================================================================
    std::cout << "\nWarmup...\n";
    for(int i = 0; i < warmup; ++i)
    {
        c_dev.zero();
        (void)dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
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
    size_t trim = times.size() / 10;
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
