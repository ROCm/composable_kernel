// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 04: Custom Heuristics
 *
 * Demonstrates custom kernel selection heuristics for different workloads.
 *
 * Build: cd dispatcher/build && cmake .. && make gemm_04_heuristics
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// KERNEL SET: Multiple tile sizes for heuristic-based selection
// =============================================================================

DECL_KERNEL_SET(heuristics_kernels,
                // Small tile - low latency
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(64, 64, 32)
                         .wave(2, 2, 1)
                         .warp(32, 32, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx942")
                    // Medium tile - balanced
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 128, 64)
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942"));

// =============================================================================
// Custom Heuristic
// =============================================================================

std::vector<std::string> size_based_heuristic(const Problem& problem)
{
    std::vector<std::string> ranked_kernels;
    int64_t total_elements = problem.M * problem.N;

    if(total_elements < 100000)
    {
        ranked_kernels = {"gemm_64x64", "gemm_128x128"};
    }
    else
    {
        ranked_kernels = {"gemm_128x128", "gemm_64x64"};
    }
    return ranked_kernels;
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 04: Custom Heuristics",
                     "Demonstrates custom kernel selection heuristics");
    args.add_option("--arch", "gfx942", "GPU architecture");

    if(!args.parse(argc, argv))
        return 0;

    print_header("Example 04: Custom Heuristics");

    std::string gfx_arch = args.get("--arch", "gfx942");

    // =========================================================================
    // Setup Registry and Dispatcher
    // =========================================================================
    Registry registry;
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);

    Dispatcher dispatcher(&registry);
    dispatcher.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
    dispatcher.set_heuristic(size_based_heuristic);

    std::cout << "\nSetup:\n";
    std::cout << "  Registry: " << registry.size() << " kernel(s)\n";
    std::cout << "  Strategy: Heuristic (size-based)\n";

    // =========================================================================
    // Test Different Problem Sizes
    // =========================================================================
    std::cout << "\nTesting heuristic selection:\n";
    print_separator();

    using DataType = ck_tile::fp16_t;

    std::vector<std::tuple<int, int, int>> sizes = {
        {128, 128, 64},
        {512, 512, 256},
        {2048, 2048, 1024},
    };

    bool all_passed = true;

    for(const auto& [M, N, K] : sizes)
    {
        Problem problem(M, N, K);
        auto selected = dispatcher.select_kernel(problem);

        std::cout << "Problem " << M << "x" << N << "x" << K << ":\n";
        if(selected)
        {
            std::cout << "  Selected: " << selected->get_name() << "\n";
        }

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

        std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
        std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";

        // Verify
        std::vector<DataType> c_host(M * N);
        c_dev.copy_to_host(c_host.data());
        float expected = static_cast<float>(K);
        int errors     = 0;
        for(int i = 0; i < M * N; ++i)
        {
            float actual = static_cast<float>(c_host[i]);
            if(std::abs(actual - expected) > 0.01f * expected + 1.0f)
                ++errors;
        }
        bool pass = (errors == 0);
        std::cout << "  Verify: " << (pass ? "PASS" : "FAIL") << "\n";
        if(!pass)
            all_passed = false;
        print_separator();
    }

    std::cout << "Overall: " << (all_passed ? "ALL PASSED" : "SOME FAILED") << "\n";

    return all_passed ? 0 : 1;
}
