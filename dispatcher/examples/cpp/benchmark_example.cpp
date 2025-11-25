// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Benchmark Example
 *
 * Comprehensive benchmarking of dispatcher GEMM performance.
 * Tests various problem sizes and reports detailed metrics.
 */

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;

#define HIP_CHECK(call)                                                   \
    do                                                                    \
    {                                                                     \
        hipError_t err = call;                                            \
        if(err != hipSuccess)                                             \
        {                                                                 \
            std::cerr << "HIP error: " << hipGetErrorString(err) << "\n"; \
            exit(1);                                                      \
        }                                                                 \
    } while(0)

struct BenchmarkResult
{
    int M, N, K;
    float min_ms;
    float max_ms;
    float avg_ms;
    float median_ms;
    float tflops;
    float bandwidth_gb;
};

KernelKey create_kernel_key()
{
    KernelKey key;
    key.signature.dtype_a             = DataType::FP16;
    key.signature.dtype_b             = DataType::FP16;
    key.signature.dtype_c             = DataType::FP16;
    key.signature.dtype_acc           = DataType::FP32;
    key.signature.layout_a            = LayoutTag::RowMajor;
    key.signature.layout_b            = LayoutTag::ColMajor;
    key.signature.layout_c            = LayoutTag::RowMajor;
    key.signature.transpose_a         = false;
    key.signature.transpose_b         = false;
    key.signature.grouped             = false;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = SelectedKernel::UseStructuredSparsity;

    key.algorithm.tile_shape.m      = SelectedKernel::TileM;
    key.algorithm.tile_shape.n      = SelectedKernel::TileN;
    key.algorithm.tile_shape.k      = SelectedKernel::TileK;
    key.algorithm.wave_shape.m      = SelectedKernel::WarpPerBlock_M;
    key.algorithm.wave_shape.n      = SelectedKernel::WarpPerBlock_N;
    key.algorithm.wave_shape.k      = SelectedKernel::WarpPerBlock_K;
    key.algorithm.warp_tile_shape.m = SelectedKernel::WarpTileM;
    key.algorithm.warp_tile_shape.n = SelectedKernel::WarpTileN;
    key.algorithm.warp_tile_shape.k = SelectedKernel::WarpTileK;
    key.algorithm.pipeline          = Pipeline::CompV4;
    key.algorithm.scheduler         = Scheduler::Intrawave;
    key.algorithm.epilogue          = Epilogue::CShuffle;
    key.algorithm.block_size        = SelectedKernel::BlockSize;
    key.algorithm.double_buffer     = SelectedKernel::DoubleSmemBuffer;
    key.algorithm.persistent        = SelectedKernel::UsePersistentKernel;
    key.algorithm.preshuffle        = SelectedKernel::Preshuffle;
    key.algorithm.transpose_c       = SelectedKernel::TransposeC;
    key.algorithm.num_wave_groups   = SelectedKernel::NumWaveGroups;
    key.gfx_arch                    = "gfx942";

    return key;
}

BenchmarkResult
benchmark_size(Dispatcher& dispatcher, int M, int N, int K, int warmup_runs, int bench_runs)
{
    Problem problem(M, N, K);

    // Allocate GPU memory
    ADataType *a_dev, *b_dev;
    CDataType* c_dev;
    HIP_CHECK(hipMalloc(&a_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&b_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&c_dev, M * N * sizeof(CDataType)));

    // Initialize with random data
    std::vector<ADataType> a_host(M * K, ADataType(1.0f));
    std::vector<BDataType> b_host(K * N, BDataType(1.0f));

    HIP_CHECK(hipMemcpy(a_dev, a_host.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_dev, b_host.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(c_dev, 0, M * N * sizeof(CDataType)));

    // Warmup
    for(int i = 0; i < warmup_runs; i++)
    {
        (void)dispatcher.run(a_dev, b_dev, c_dev, problem, nullptr);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    std::vector<float> times;
    times.reserve(bench_runs);

    for(int i = 0; i < bench_runs; i++)
    {
        float time_ms = dispatcher.run(a_dev, b_dev, c_dev, problem, nullptr);
        times.push_back(time_ms);
    }

    // Cleanup
    HIP_CHECK(hipFree(a_dev));
    HIP_CHECK(hipFree(b_dev));
    HIP_CHECK(hipFree(c_dev));

    // Compute statistics
    std::sort(times.begin(), times.end());

    float min_ms    = times.front();
    float max_ms    = times.back();
    float avg_ms    = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
    float median_ms = times[times.size() / 2];

    // Performance metrics
    double flops = 2.0 * M * N * K;
    float tflops = flops / (min_ms * 1e9);

    // Memory bandwidth (approximation)
    double bytes       = (M * K + K * N + M * N) * sizeof(ADataType);
    float bandwidth_gb = bytes / (min_ms * 1e6);

    return {M, N, K, min_ms, max_ms, avg_ms, median_ms, tflops, bandwidth_gb};
}

void print_results(const std::vector<BenchmarkResult>& results)
{
    std::cout << "\n";
    std::cout << std::setw(20) << "Size" << std::setw(12) << "Min (ms)" << std::setw(12)
              << "Avg (ms)" << std::setw(12) << "Med (ms)" << std::setw(12) << "Max (ms)"
              << std::setw(12) << "TFLOPS" << std::setw(12) << "BW (GB/s)" << "\n";
    std::cout << std::string(92, '-') << "\n";

    for(const auto& r : results)
    {
        std::ostringstream size_str;
        size_str << r.M << "x" << r.N << "x" << r.K;

        std::cout << std::setw(20) << size_str.str() << std::setw(12) << std::fixed
                  << std::setprecision(4) << r.min_ms << std::setw(12) << std::fixed
                  << std::setprecision(4) << r.avg_ms << std::setw(12) << std::fixed
                  << std::setprecision(4) << r.median_ms << std::setw(12) << std::fixed
                  << std::setprecision(4) << r.max_ms << std::setw(12) << std::fixed
                  << std::setprecision(2) << r.tflops << std::setw(12) << std::fixed
                  << std::setprecision(2) << r.bandwidth_gb << "\n";
    }
}

int main(int argc, char** argv)
{
    std::cout << "======================================================================\n";
    std::cout << "CK Tile Dispatcher - Benchmark Example\n";
    std::cout << "======================================================================\n\n";

    // GPU info
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (" << prop.gcnArchName << ")\n";
    std::cout << "Kernel: " << KERNEL_NAME << "\n\n";

    // Register kernel
    auto key = create_kernel_key();
    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Registry::Priority::High);

    Dispatcher dispatcher;

    // Benchmark configuration
    const int warmup_runs = 3;
    const int bench_runs  = 10;

    std::cout << "Configuration:\n";
    std::cout << "  Warmup runs: " << warmup_runs << "\n";
    std::cout << "  Benchmark runs: " << bench_runs << "\n";

    // Test sizes
    std::vector<std::tuple<int, int, int>> sizes = {
        // Square sizes
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},

        // Rectangular sizes
        {512, 512, 2048},
        {512, 2048, 512},
        {2048, 512, 512},

        // Common deep learning sizes
        {1024, 4096, 1024},
        {4096, 1024, 1024},
        {1024, 1024, 4096},
    };

    std::cout << "\nRunning benchmarks...\n";

    std::vector<BenchmarkResult> results;
    for(const auto& [M, N, K] : sizes)
    {
        std::cout << "  " << M << "x" << N << "x" << K << "..." << std::flush;
        auto result = benchmark_size(dispatcher, M, N, K, warmup_runs, bench_runs);
        results.push_back(result);
        std::cout << " " << result.tflops << " TFLOPS\n";
    }

    // Print results
    print_results(results);

    // Summary
    float max_tflops = 0;
    for(const auto& r : results)
    {
        max_tflops = std::max(max_tflops, r.tflops);
    }

    std::cout << "\n======================================================================\n";
    std::cout << "Peak Performance: " << max_tflops << " TFLOPS\n";
    std::cout << "======================================================================\n";

    return 0;
}
