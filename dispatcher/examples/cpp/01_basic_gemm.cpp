// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 01: Basic GEMM with KernelSet
 *
 * Demonstrates the declarative kernel specification with explicit
 * Signature/Algorithm structs. All kernel key-values are visible.
 *
 * Build:
 *   python3 scripts/build_with_kernels.py examples/cpp/01_basic_gemm.cpp
 *
 * Complexity: ★☆☆☆☆
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
// KERNEL SET DECLARATIONS
// =============================================================================

// -----------------------------------------------------------------------------
// Kernel set with FULL explicit configuration
// All parameters visible: dtype, layout, tile, wave, warp, pipeline, etc.
// -----------------------------------------------------------------------------
DECL_KERNEL_SET(explicit_config,
                .add(Signature()
                         .dtype("fp16", "fp16", "fp16", "fp32") // A, B, C, Accumulator
                         .layout("row", "col", "row"),          // A=row, B=col, C=row
                     Algorithm()
                         .tile(128, 128, 32)     // Block tile: M, N, K
                         .wave(2, 2, 1)          // Warps per block
                         .warp(32, 32, 16)       // Warp tile
                         .pipeline("compv4")     // Pipeline type
                         .scheduler("intrawave") // Scheduler
                         .epilogue("cshuffle")   // Epilogue
                         .pad(true, true, true)) // Padding M, N, K
);

// -----------------------------------------------------------------------------
// Kernel set with COMPACT syntax
// Unspecified values auto-expand to all valid combinations
// -----------------------------------------------------------------------------
DECL_KERNEL_SET(auto_expand,
                .add("fp16", "rcr", 64, 64, 32)       // wave/warp auto-expand
                    .add("fp16", "rcr", 256, 256, 64) // generates all valid combos
);

// -----------------------------------------------------------------------------
// Kernel set with MIXED data types
// -----------------------------------------------------------------------------
DECL_KERNEL_SET(mixed_dtypes, .add("fp16", "rcr", 128, 128, 32).add("bf16", "rcr", 128, 128, 32));

// -----------------------------------------------------------------------------
// Kernel set with DIFFERENT layouts
// -----------------------------------------------------------------------------
DECL_KERNEL_SET(layouts,
                .add("fp16", "rcr", 128, 128, 32)     // Row-Col-Row (BLAS-style)
                    .add("fp16", "rrr", 128, 128, 32) // All row-major
);

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    if(argc > 1 && std::string(argv[1]) == "--list")
    {
        KernelSetRegistry::instance().print();
        return 0;
    }

    print_header("Example 01: Basic GEMM");

    // =========================================================================
    // Step 1: Show all declared kernel sets
    // =========================================================================
    std::cout << "\nStep 1: Declared Kernel Sets\n";
    KernelSetRegistry::instance().print();

    // =========================================================================
    // Step 2: Create Registry
    // =========================================================================
    std::cout << "\nStep 2: Create Registry\n";
    Registry registry;
    registry.set_name("declarative_registry");

    KernelConfig config =
        KernelConfig::fp16_rcr()
            .tile(SelectedKernel::TileM, SelectedKernel::TileN, SelectedKernel::TileK)
            .wave(SelectedKernel::WarpPerBlock_M,
                  SelectedKernel::WarpPerBlock_N,
                  SelectedKernel::WarpPerBlock_K)
            .warp_tile(
                SelectedKernel::WarpTileM, SelectedKernel::WarpTileN, SelectedKernel::WarpTileK)
            .block(SelectedKernel::BlockSize);

    KernelKey key = config.build_key();
    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, KERNEL_NAME);

    registry.register_kernel(kernel);
    std::cout << "  Registered: " << kernel->get_name() << "\n";

    // =========================================================================
    // Step 3: Create Dispatcher and Run
    // =========================================================================
    std::cout << "\nStep 3: Run GEMM\n";
    Dispatcher dispatcher(&registry);

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

    auto selected = dispatcher.select_kernel(problem);
    std::cout << "  Problem: " << M << " x " << N << " x " << K << "\n";
    std::cout << "  Kernel:  " << selected->get_name() << "\n";

    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
    std::cout << "  Time:    " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS:  " << std::setprecision(2) << calculate_tflops(M, N, K, time_ms)
              << "\n";

    // =========================================================================
    // Step 4: Verify
    // =========================================================================
    std::cout << "\nStep 4: Verify\n";
    std::vector<CDataType> c_host(M * N);
    c_dev.copy_to_host(c_host.data());

    float expected = static_cast<float>(K);
    float actual   = static_cast<float>(c_host[0]);
    bool passed    = std::abs(actual - expected) < 1.0f;

    std::cout << "  C[0,0] = " << actual << " (expected " << expected << ")\n";
    std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n";

    // =========================================================================
    // Summary
    // =========================================================================
    print_separator();
    std::cout << "Full Declaration Syntax:\n";
    print_separator();
    std::cout << "DECL_KERNEL_SET(my_kernels,\n";
    std::cout << "    .add(Signature()\n";
    std::cout << "            .dtype(\"fp16\", \"fp16\", \"fp16\", \"fp32\")\n";
    std::cout << "            .layout(\"row\", \"col\", \"row\"),\n";
    std::cout << "         Algorithm()\n";
    std::cout << "            .tile(128, 128, 32)\n";
    std::cout << "            .wave(2, 2, 1)\n";
    std::cout << "            .warp(32, 32, 16)\n";
    std::cout << "            .pipeline(\"compv4\")\n";
    std::cout << "            .scheduler(\"intrawave\")\n";
    std::cout << "            .epilogue(\"cshuffle\")\n";
    std::cout << "            .pad(true, true, true))\n";
    std::cout << ");\n";
    print_separator();

    return passed ? 0 : 1;
}
