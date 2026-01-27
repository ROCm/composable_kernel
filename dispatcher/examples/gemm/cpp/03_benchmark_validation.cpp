// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 03: GEMM Benchmark & Validation
 *
 * Combined example demonstrating:
 *   1. Benchmarking with statistics (warmup, iterations, min/max/mean/median)
 *   2. Validation against CK Tile reference (CPU or GPU)
 *
 * Build: cd dispatcher/build && cmake .. && make gemm_03_benchmark_validation
 * Usage: ./gemm_03_benchmark_validation [--size N] [--verify MODE] [--benchmark]
 *
 * Options:
 *   --size N        Problem size MxNxK (default: 512)
 *   --verify MODE   0=none, 1=CPU ref, 2=GPU ref (default: 1)
 *   --benchmark     Run full benchmark with statistics
 *   --warmup N      Warmup iterations (default: 5)
 *   --iterations N  Benchmark iterations (default: 20)
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;
using namespace ck_tile::literals;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// KERNEL SET: High-performance kernels for benchmarking/validation
// =============================================================================

DECL_KERNEL_SET(benchmark_validation_kernels,
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(128, 128, 32)
                         .wave(2, 2, 1)
                         .warp(32, 32, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx942"));

// =============================================================================
// Helper: Layout detection
// =============================================================================

template <typename Layout>
constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 03: GEMM Benchmark & Validation",
                     "Benchmark and/or validate GEMM output against reference");
    args.add_option("--size", "512", "Problem size MxNxK");
    args.add_option("--verify", "1", "Verification: 0=none, 1=CPU ref, 2=GPU ref");
    args.add_flag("--benchmark", "Run benchmark with statistics");
    args.add_option("--warmup", "5", "Warmup iterations");
    args.add_option("--iterations", "20", "Benchmark iterations");
    args.add_option("--rtol", "0.01", "Relative tolerance");
    args.add_option("--atol", "0.01", "Absolute tolerance");
    args.add_option("--arch", "gfx942", "GPU architecture");

    if(!args.parse(argc, argv))
        return 0;

    int M                = args.get_int("--size", 512);
    int N                = M;
    int K                = M;
    int verify           = args.get_int("--verify", 1);
    bool do_benchmark    = args.has("--benchmark");
    int warmup           = args.get_int("--warmup", 5);
    int iterations       = args.get_int("--iterations", 20);
    float rtol           = args.get_float("--rtol", 0.01f);
    float atol           = args.get_float("--atol", 0.01f);
    std::string gfx_arch = args.get("--arch", "gfx942");

    print_header("Example 03: GEMM Benchmark & Validation");

    std::cout << "\nConfiguration:\n";
    std::cout << "  Problem:     " << M << " x " << N << " x " << K << "\n";
    std::cout << "  Layout:      RCR (A=row, B=col, C=row)\n";
    std::cout << "  Verify:      " << verify;
    if(verify == 0)
        std::cout << " (disabled)";
    else if(verify == 1)
        std::cout << " (CPU reference)";
    else if(verify == 2)
        std::cout << " (GPU reference)";
    std::cout << "\n";
    std::cout << "  Benchmark:   " << (do_benchmark ? "yes" : "no") << "\n";
    if(do_benchmark)
    {
        std::cout << "    Warmup:    " << warmup << " iterations\n";
        std::cout << "    Measure:   " << iterations << " iterations\n";
    }

    // =========================================================================
    // Setup Registry and Dispatcher
    // =========================================================================
    Registry registry;
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    Dispatcher dispatcher(&registry);

    std::cout << "  Kernels:     " << registry.size() << " registered\n";
    print_registered_kernels(registry);

    // =========================================================================
    // Initialize data with proper tensor descriptors
    // =========================================================================
    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using ADataType   = ck_tile::fp16_t;
    using BDataType   = ck_tile::fp16_t;
    using CDataType   = ck_tile::fp16_t;
    using AccDataType = float;

    auto stride_a = ck_tile::get_default_stride(M, K, 0_uz, is_row_major(ALayout{}));
    auto stride_b = ck_tile::get_default_stride(K, N, 0_uz, is_row_major(BLayout{}));
    auto stride_c = ck_tile::get_default_stride(M, N, 0_uz, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_a, is_row_major(ALayout{})));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_b, is_row_major(BLayout{})));
    ck_tile::HostTensor<CDataType> c_m_n_dev(
        ck_tile::host_tensor_descriptor(M, N, stride_c, is_row_major(CLayout{})));
    ck_tile::HostTensor<CDataType> c_m_n_ref(
        ck_tile::host_tensor_descriptor(M, N, stride_c, is_row_major(CLayout{})));

    ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_m_k);
    ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_k_n);

    std::cout << "\nData:\n";
    std::cout << "  A: " << M << " x " << K << " (fp16, row-major)\n";
    std::cout << "  B: " << K << " x " << N << " (fp16, col-major)\n";
    std::cout << "  C: " << M << " x " << N << " (fp16, row-major)\n";

    // GPU memory
    ck_tile::DeviceMem a_dev(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev(c_m_n_dev.get_element_space_size_in_bytes());

    a_dev.ToDevice(a_m_k.data());
    b_dev.ToDevice(b_k_n.data());

    // =========================================================================
    // Compute Reference (if needed)
    // =========================================================================
    if(verify > 0)
    {
        std::cout << "\nComputing reference...\n";
        c_m_n_ref.SetZero();

        if(verify == 1)
        {
            std::cout << "  Using CPU reference (ck_tile::reference_gemm)\n";
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_m_k, b_k_n, c_m_n_ref);
        }
        else if(verify == 2)
        {
            std::cout << "  Using GPU reference (ck_tile::reference_gemm_gpu)\n";
            ck_tile::DeviceMem c_ref_dev(c_m_n_ref.get_element_space_size_in_bytes());
            c_ref_dev.SetZero();

            ck_tile::reference_gemm_gpu<ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        CLayout>(
                static_cast<ADataType*>(a_dev.GetDeviceBuffer()),
                static_cast<BDataType*>(b_dev.GetDeviceBuffer()),
                static_cast<CDataType*>(c_ref_dev.GetDeviceBuffer()),
                M,
                N,
                K,
                stride_a,
                stride_b,
                stride_c);

            (void)hipDeviceSynchronize();
            c_ref_dev.FromDevice(c_m_n_ref.data());
        }
        std::cout << "  Reference complete.\n";
    }

    // =========================================================================
    // Run Kernel
    // =========================================================================
    Problem problem(M, N, K);
    auto selected = dispatcher.select_kernel(problem);

    std::cout << "\nRunning kernel:\n";
    if(selected)
        std::cout << "  Selected: " << selected->get_name() << "\n";

    c_dev.SetZero();
    float time_ms = 0.0f;
    std::vector<float> times;

    if(do_benchmark)
    {
        // Warmup
        std::cout << "  Warming up (" << warmup << " iterations)...\n";
        for(int i = 0; i < warmup; ++i)
        {
            c_dev.SetZero();
            (void)dispatcher.run(static_cast<ADataType*>(a_dev.GetDeviceBuffer()),
                                 static_cast<BDataType*>(b_dev.GetDeviceBuffer()),
                                 static_cast<CDataType*>(c_dev.GetDeviceBuffer()),
                                 problem,
                                 nullptr);
        }

        // Benchmark
        std::cout << "  Benchmarking (" << iterations << " iterations)...\n";
        times.reserve(iterations);
        for(int i = 0; i < iterations; ++i)
        {
            c_dev.SetZero();
            float t = dispatcher.run(static_cast<ADataType*>(a_dev.GetDeviceBuffer()),
                                     static_cast<BDataType*>(b_dev.GetDeviceBuffer()),
                                     static_cast<CDataType*>(c_dev.GetDeviceBuffer()),
                                     problem,
                                     nullptr);
            times.push_back(t);
        }
        time_ms = *std::min_element(times.begin(), times.end());
    }
    else
    {
        // Single run
        time_ms = dispatcher.run(static_cast<ADataType*>(a_dev.GetDeviceBuffer()),
                                 static_cast<BDataType*>(b_dev.GetDeviceBuffer()),
                                 static_cast<CDataType*>(c_dev.GetDeviceBuffer()),
                                 problem,
                                 nullptr);
    }

    c_dev.FromDevice(c_m_n_dev.data());

    // =========================================================================
    // Results
    // =========================================================================
    double flops  = 2.0 * M * N * K;
    double tflops = flops / (time_ms * 1e9);

    print_separator();
    std::cout << "Performance:\n";
    print_separator();

    if(do_benchmark && !times.empty())
    {
        std::sort(times.begin(), times.end());
        float min_t    = times.front();
        float max_t    = times.back();
        float median_t = times[times.size() / 2];
        float mean_t   = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Min:      " << min_t << " ms (" << std::setprecision(2)
                  << (flops / (min_t * 1e9)) << " TFLOPS)\n";
        std::cout << std::setprecision(4);
        std::cout << "  Max:      " << max_t << " ms\n";
        std::cout << "  Mean:     " << mean_t << " ms (" << std::setprecision(2)
                  << (flops / (mean_t * 1e9)) << " TFLOPS)\n";
        std::cout << std::setprecision(4);
        std::cout << "  Median:   " << median_t << " ms (" << std::setprecision(2)
                  << (flops / (median_t * 1e9)) << " TFLOPS)\n";
    }
    else
    {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Time:     " << time_ms << " ms\n";
        std::cout << std::setprecision(2);
        std::cout << "  TFLOPS:   " << tflops << "\n";
    }

    // =========================================================================
    // Validation
    // =========================================================================
    bool pass = true;

    if(verify > 0)
    {
        print_separator();
        std::cout << "Validation:\n";
        print_separator();
        std::cout << "  Tolerance: rtol=" << rtol << ", atol=" << atol << "\n";

        pass = ck_tile::check_err(c_m_n_dev, c_m_n_ref, "Validation Error!", rtol, atol);

        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        for(size_t i = 0; i < c_m_n_dev.get_element_space_size(); ++i)
        {
            float dev_val  = static_cast<float>(c_m_n_dev.mData[i]);
            float ref_val  = static_cast<float>(c_m_n_ref.mData[i]);
            float abs_diff = std::abs(dev_val - ref_val);
            float rel_diff = (ref_val != 0.0f) ? abs_diff / std::abs(ref_val) : abs_diff;
            max_abs_diff   = std::max(max_abs_diff, abs_diff);
            max_rel_diff   = std::max(max_rel_diff, rel_diff);
        }

        std::cout << "  Max abs diff: " << max_abs_diff << "\n";
        std::cout << "  Max rel diff: " << max_rel_diff << "\n";
    }

    // =========================================================================
    // Summary
    // =========================================================================
    print_separator();
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << "\n";
    print_separator();

    return pass ? 0 : 1;
}
