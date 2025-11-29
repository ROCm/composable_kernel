// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 07: Preshuffle GEMM
 *
 * Demonstrates weight preshuffling for inference workloads.
 *
 * Build:
 *   python3 scripts/compile_gemm_examples.py examples/cpp/07_preshuffle.cpp
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
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// KERNEL SET: Preshuffle-optimized kernels
// =============================================================================

DECL_KERNEL_SET(preshuffle,
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm().tile(128, 128, 32).preshuffle(true)) // Enable weight preshuffle
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm().tile(256, 256, 64).preshuffle(true)));

// Standard kernels for comparison
DECL_KERNEL_SET(standard, .add("fp16", "rcr", 128, 128, 32));

// =============================================================================
// MAIN
// =============================================================================

int main()
{
    print_header("Example 07: Preshuffle GEMM");

    std::cout << "\nPreshuffle Benefits:\n";
    std::cout << "  - Weight matrix is pre-transformed offline\n";
    std::cout << "  - Faster inference (weights are fixed)\n";
    std::cout << "  - Optimized memory access patterns\n";

    // =========================================================================
    // Setup
    // =========================================================================
    std::cout << "\nSetup:\n";
    Registry registry;
    registry.set_name("preshuffle_registry");

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

    std::cout << "  Kernel: " << kernel->get_name() << "\n";

    // =========================================================================
    // Run GEMM
    // =========================================================================
    const int M = 2048, N = 2048, K = 1024;
    Problem problem(M, N, K);

    GpuBuffer<ADataType> a_dev(M * K);
    GpuBuffer<BDataType> b_dev(K * N);
    GpuBuffer<CDataType> c_dev(M * N);

    std::vector<ADataType> a_host(M * K, ADataType(1.0f));
    std::vector<BDataType> b_host(K * N, BDataType(1.0f));
    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());
    c_dev.zero();

    std::cout << "\nRunning GEMM (" << M << " x " << N << " x " << K << ")...\n";
    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);

    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << calculate_tflops(M, N, K, time_ms) << "\n";

    // =========================================================================
    // Verify
    // =========================================================================
    std::vector<CDataType> c_host(M * N);
    c_dev.copy_to_host(c_host.data());

    float expected = static_cast<float>(K);
    float actual   = static_cast<float>(c_host[0]);
    bool passed    = std::abs(actual - expected) < 1.0f;

    print_separator();
    std::cout << "Result: C[0,0] = " << actual << " (expected " << expected << ")\n";
    std::cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    return passed ? 0 : 1;
}
