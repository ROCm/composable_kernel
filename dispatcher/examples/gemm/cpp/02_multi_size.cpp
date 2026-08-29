// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 02: Multi-Size GEMM with Wildcard Expansion
 *
 * Demonstrates the WILDCARD feature where specifying wildcards causes
 * the build system to expand to ALL valid configurations for the architecture.
 *
 * WILDCARD SYNTAX:
 *   - Integer params: ANY_INT or -1 (both are equivalent, ANY_INT is just a #define for -1)
 *   - String params:  "*" (for pipeline, scheduler)
 *
 * The kernel declaration:
 *   .add(..., Algorithm().tile(64,64,64).wave(ANY_INT,ANY_INT,1).warp(-1,-1,-1)
 *                        .pipeline("*").scheduler("*"), ...)
 *
 * Expands to multiple kernels:
 *   - wave: (1,4,1), (2,2,1), (4,1,1)  -> 3 options
 *   - warp: (16,16,32), (32,32,16)     -> 2 options
 *   - pipeline: "compv3"               -> 1 option (compv4 requires special handling)
 *   - scheduler: "intrawave"           -> 1 option
 *
 * Raw expansion: 3 x 2 = 6 configs, but arch filter validates each:
 *   - tile_m must be divisible by (warp_m x warp_tile_m)
 *   - tile_n must be divisible by (warp_n x warp_tile_n)
 *   - Some wave/warp combos invalid: (4,1,1)+(32,32,16), (1,4,1)+(32,32,16)
 * Result: 4 valid wildcard kernels + 1 explicit = 5 total
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
                    // Wildcards: ANY_INT == -1 (for integers), "*" (for strings)
                    // -------------------------------------------------------------------------
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(64, 64, 64)
                             .wave(ANY_INT, ANY_INT, 1) // ANY_INT -> (1,4,1), (2,2,1), (4,1,1)
                             .warp(-1, -1, -1) // -1 same as ANY_INT -> (16,16,32), (32,32,16)
                             .pipeline("*")    // "*" -> valid pipelines
                             .scheduler("*")   // "*" -> valid schedulers
                             .epilogue("cshuffle"),
                         "gfx942"));
// Raw: 3x2=6, arch filter removes 2 invalid -> 4 valid kernels

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
    args.add_flag("--list-verbose", "List kernels with full configuration details");

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
  Wildcard syntax:
    ANY_INT or -1  -> expands integer params to all valid values
    "*"            -> expands string params (pipeline/scheduler) to valid values

  Declaration with wildcards:
    .tile(64, 64, 64)           -> fixed tile size (no wildcard)
    .wave(ANY_INT, ANY_INT, 1)  -> expands to (1,4,1), (2,2,1), (4,1,1) = 3
    .warp(-1, -1, -1)           -> expands to (16,16,32), (32,32,16) = 2
    .pipeline("*")              -> expands to valid pipelines = 1
    .scheduler("*")             -> expands to valid schedulers = 1

  Expanded: 3 x 2 = 6 configs, but arch filter validates each:
    - wave x warp must divide tile: (4,1,1)x(32,32,16) invalid for 64x64
    - Result: 4 valid kernels from wildcard + 1 explicit = 5 total
)";

    // =========================================================================
    // Setup Registry and Dispatcher
    // =========================================================================
    std::cout << "\nStep 1: Register Kernels\n";
    std::cout << "------------------------\n";

    Registry registry;
    registry.set_name("multi_size_registry");

    // Register kernels from generated header (includes expanded wildcards)
    // Use generic macro - no need to hardcode example name
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s) from wildcard expansion\n";

    if(args.has("--list") || args.has("--list-verbose"))
    {
        std::cout << "\n";
        print_registered_kernels(registry, std::cout, args.has("--list-verbose"));
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
