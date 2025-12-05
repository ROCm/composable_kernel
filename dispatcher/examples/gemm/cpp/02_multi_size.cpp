// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 02: Multi-Size GEMM with Wildcard Expansion
 *
 * Demonstrates the WILDCARD feature where specifying ANY_INT or "*" causes
 * the build system to expand to ALL valid configurations for the architecture.
 *
 * The kernel declaration:
 *   .add(..., Algorithm().tile(128,128,64).wave(ANY_INT,ANY_INT,1).warp(ANY_INT,ANY_INT,16)
 *                        .pipeline("*").scheduler("*"), ...)
 *
 * Expands to multiple kernels:
 *   - wave: (1,4,1), (2,2,1), (4,1,1)  -> 3 options
 *   - warp: (32,32,16), (16,16,32)     -> 2 options
 *   - pipeline: "compv3"               -> 1 option (compv4 requires special handling)
 *   - scheduler: "intrawave"           -> 1 option
 *
 * Build: cd dispatcher/build && cmake .. && make gemm_02_multi_size
 * Usage: ./gemm_02_multi_size [--max-size N] [--help]
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// KERNEL SET: Demonstrates Wildcard Expansion
// =============================================================================

DECL_KERNEL_SET(multi_size_kernels,
                // -------------------------------------------------------------------------
                // Kernel 1: Explicit - all parameters specified (no expansion)
                // -------------------------------------------------------------------------
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(64, 64, 32)
                         .wave(2, 2, 1)
                         .warp(16, 16, 32)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx942")

                    // -------------------------------------------------------------------------
                    // Kernel 2: WILDCARD - expands to multiple valid configurations
                    // ANY_INT and "*" are expanded by build system to all arch-valid combos
                    // -------------------------------------------------------------------------
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 128, 64)
                             .wave(ANY_INT, ANY_INT, 1)  // Expands to: (1,4,1), (2,2,1), (4,1,1)
                             .warp(ANY_INT, ANY_INT, 16) // Expands to: (32,32,16), (16,16,32)
                             .pipeline("*")              // Expands to valid pipelines
                             .scheduler("*")             // Expands to valid schedulers
                             .epilogue("cshuffle"),
                         "gfx942"));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 02: Multi-Size GEMM with Wildcards",
                     "Demonstrates wildcard expansion for kernel generation");
    args.add_option("--max-size", "4096", "Maximum problem size to test");
    args.add_option("--arch", "gfx942", "GPU architecture");
    args.add_flag("--list", "List all registered kernels");

    if(!args.parse(argc, argv))
        return 0;

    int max_size         = args.get_int("--max-size", 4096);
    std::string gfx_arch = args.get("--arch", "gfx942");

    print_header("Example 02: Multi-Size GEMM with Wildcards");

    // =========================================================================
    // Show Wildcard Expansion Concept
    // =========================================================================
    std::cout << "\nWILDCARD EXPANSION:\n";
    std::cout << "===================\n";
    std::cout << R"(
  Declaration with wildcards:
    .tile(128, 128, 64)
    .wave(ANY_INT, ANY_INT, 1)  -> expands to (1,4,1), (2,2,1), (4,1,1)
    .warp(ANY_INT, ANY_INT, 16) -> expands to (32,32,16), (16,16,32)
    .pipeline("*")              -> expands to valid pipelines
    .scheduler("*")             -> expands to valid schedulers

  This generates multiple kernels from ONE declaration!
)";

    // =========================================================================
    // Setup Registry and Dispatcher
    // =========================================================================
    std::cout << "\nStep 1: Register Kernels\n";
    std::cout << "------------------------\n";

    Registry registry;
    registry.set_name("multi_size_registry");

    // Register kernels from generated header (includes expanded wildcards)
    generated::register_02_multi_size_kernels(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s) from wildcard expansion\n";

    if(args.has("--list"))
    {
        std::cout << "\n  Available kernels:\n";
        for(const auto& k : registry.get_all())
        {
            std::cout << "    - " << k->get_name() << "\n";
        }
        return 0;
    }

    Dispatcher dispatcher(&registry);
    std::cout << "  Max size: " << max_size << "\n";

    // =========================================================================
    // Run Multiple Problem Sizes
    // =========================================================================
    std::cout << "\nStep 2: Run Multiple Sizes\n";
    print_separator();
    std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "K"
              << std::setw(12) << "Time(ms)" << std::setw(12) << "TFLOPS" << "\n";
    print_separator();

    std::vector<std::tuple<int, int, int>> all_sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
    };

    std::vector<std::tuple<int, int, int>> sizes;
    for(const auto& [M, N, K] : all_sizes)
    {
        if(std::max({M, N, K}) <= max_size)
            sizes.push_back({M, N, K});
    }

    using DataType  = ck_tile::fp16_t;
    bool all_passed = true;

    for(const auto& [M, N, K] : sizes)
    {
        Problem problem(M, N, K);

        GpuBuffer<DataType> a_dev(M * K);
        GpuBuffer<DataType> b_dev(K * N);
        GpuBuffer<DataType> c_dev(M * N);

        std::vector<DataType> a_host(M * K, DataType(1.0f));
        std::vector<DataType> b_host(K * N, DataType(1.0f));
        a_dev.copy_from_host(a_host.data());
        b_dev.copy_from_host(b_host.data());
        c_dev.zero();

        float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
        double tflops = calculate_tflops(M, N, K, time_ms);

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << K << std::setw(12)
                  << std::fixed << std::setprecision(4) << time_ms << std::setw(12)
                  << std::setprecision(2) << tflops << "\n";

        // Verify
        std::vector<DataType> c_host(M * N);
        c_dev.copy_to_host(c_host.data());
        float expected = static_cast<float>(K);
        int errors     = 0;
        for(int i = 0; i < M * N; ++i)
        {
            if(std::abs(static_cast<float>(c_host[i]) - expected) > 0.01f * expected + 1.0f)
                ++errors;
        }
        if(errors > 0)
            all_passed = false;
    }

    print_separator();
    std::cout << "Status: " << (all_passed ? "ALL PASSED" : "SOME FAILED") << "\n";
    print_separator();

    return all_passed ? 0 : 1;
}
