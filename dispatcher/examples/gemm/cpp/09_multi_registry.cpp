// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 09: Multiple Registries
 *
 * Demonstrates using separate registries for different workload types,
 * each with its own optimized kernel set.
 *
 * Build:
 *   python3 scripts/compile_gemm_examples.py examples/cpp/09_multi_registry.cpp
 *
 * Complexity: ★★★★☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// KERNEL SETS: Different sets for different workload types
// =============================================================================

// Compute-bound: Large tiles for high arithmetic intensity
DECL_KERNEL_SET(compute_bound,
                .add("fp16", "rcr", 256, 256, 64)
                    .add("fp16", "rcr", 256, 128, 64)
                    .add("fp16", "rcr", 128, 256, 64));

// Memory-bound: Small tiles for better memory efficiency
DECL_KERNEL_SET(memory_bound,
                .add("fp16", "rcr", 64, 64, 32)
                    .add("fp16", "rcr", 64, 128, 32)
                    .add("fp16", "rcr", 128, 64, 32));

// Latency-optimized: Minimal tiles for low latency
DECL_KERNEL_SET(latency_opt, .add("fp16", "rcr", 32, 32, 16).add("fp16", "rcr", 64, 64, 16));

// BF16 workloads
DECL_KERNEL_SET(bf16_compute, .add("bf16", "rcr", 128, 128, 32).add("bf16", "rcr", 256, 256, 64));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 09: Multiple Registries",
                     "Separate registries for different workload types");
    args.add_flag("--list", "List all kernel sets");

    if(!args.parse(argc, argv))
        return 0;

    print_header("Example 09: Multiple Registries");

    if(args.has("--list"))
    {
        std::cout << "\nDeclared Kernel Sets:\n";
        KernelSetRegistry::instance().print();
        return 0;
    }

    // =========================================================================
    // Show declared kernel sets
    // =========================================================================
    std::cout << "\nDeclared Kernel Sets:\n";
    KernelSetRegistry::instance().print();

    // =========================================================================
    // Create separate registries
    // =========================================================================
    std::cout << "\nCreating specialized registries...\n";

    // In a real scenario, each registry would have different kernels loaded
    // For this demo, we use the same generated kernel
    Registry compute_registry;
    Registry memory_registry;
    Registry latency_registry;

    compute_registry.set_name("compute_bound");
    memory_registry.set_name("memory_bound");
    latency_registry.set_name("latency_optimized");

    // Add the generated kernel to all registries (demo)
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

    compute_registry.register_kernel(kernel);
    memory_registry.register_kernel(kernel);
    latency_registry.register_kernel(kernel);

    std::cout << "  " << compute_registry.get_name() << ": " << compute_registry.size()
              << " kernel(s)\n";
    std::cout << "  " << memory_registry.get_name() << ": " << memory_registry.size()
              << " kernel(s)\n";
    std::cout << "  " << latency_registry.get_name() << ": " << latency_registry.size()
              << " kernel(s)\n";

    // =========================================================================
    // Create dispatchers for each registry
    // =========================================================================
    Dispatcher compute_dispatcher(&compute_registry);
    Dispatcher memory_dispatcher(&memory_registry);
    Dispatcher latency_dispatcher(&latency_registry);

    // =========================================================================
    // Run with different dispatchers
    // =========================================================================
    std::cout << "\nRunning with different dispatchers:\n";
    print_separator();

    struct WorkloadTest
    {
        const char* name;
        Dispatcher* dispatcher;
        int M, N, K;
    };

    std::vector<WorkloadTest> tests = {
        {"Compute-bound", &compute_dispatcher, 4096, 4096, 4096},
        {"Memory-bound", &memory_dispatcher, 1024, 1024, 1024},
        {"Latency-opt", &latency_dispatcher, 512, 512, 512},
    };

    // Tolerance parameters for correctness check
    // With A=1, B=1: C[i,j] = K (exact for FP16 when K < 2048)
    constexpr float atol = 0.0f; // Absolute tolerance (exact match expected)
    constexpr float rtol = 0.0f; // Relative tolerance (exact match expected)

    bool all_passed = true;

    for(const auto& test : tests)
    {
        Problem problem(test.M, test.N, test.K);

        GpuBuffer<ADataType> a_dev(test.M * test.K);
        GpuBuffer<BDataType> b_dev(test.K * test.N);
        GpuBuffer<CDataType> c_dev(test.M * test.N);

        std::vector<ADataType> a_host(test.M * test.K, ADataType(1.0f));
        std::vector<BDataType> b_host(test.K * test.N, BDataType(1.0f));
        a_dev.copy_from_host(a_host.data());
        b_dev.copy_from_host(b_host.data());
        c_dev.zero();

        float time_ms =
            test.dispatcher->run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
        double tflops = calculate_tflops(test.M, test.N, test.K, time_ms);

        std::cout << test.name << " (" << test.M << "x" << test.N << "x" << test.K << "):\n";
        std::cout << "  Time:   " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
        std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";

        // Verify ALL elements using configurable tolerances
        std::vector<CDataType> c_host(test.M * test.N);
        c_dev.copy_to_host(c_host.data());
        const float expected = static_cast<float>(test.K);
        const float tol      = atol + rtol * std::abs(expected);

        int num_errors  = 0;
        float max_error = 0.0f;
        for(int i = 0; i < test.M * test.N; ++i)
        {
            float actual = static_cast<float>(c_host[i]);
            float error  = std::abs(actual - expected);
            if(error > max_error)
                max_error = error;
            if(error > tol)
                ++num_errors;
        }

        bool test_passed = (num_errors == 0);
        std::cout << "  Verify: " << (test.M * test.N) << " elements, " << "errors=" << num_errors
                  << ", max_err=" << max_error << "\n";
        std::cout << "  Status: " << (test_passed ? "PASS" : "FAIL") << "\n\n";

        if(!test_passed)
            all_passed = false;
    }

    print_separator();
    std::cout << "Multi-Registry Pattern:\n";
    print_separator();
    std::cout << "// Declare specialized kernel sets\n";
    std::cout << "DECL_KERNEL_SET(compute_bound, .add(\"fp16\", \"rcr\", 256, 256, 64));\n";
    std::cout << "DECL_KERNEL_SET(memory_bound,  .add(\"fp16\", \"rcr\", 64, 64, 32));\n";
    std::cout << "\n";
    std::cout << "// Create separate registries and dispatchers\n";
    std::cout << "Registry compute_reg, memory_reg;\n";
    std::cout << "Dispatcher compute_disp(&compute_reg);\n";
    std::cout << "Dispatcher memory_disp(&memory_reg);\n";
    std::cout << "\n";
    std::cout << "// Choose dispatcher based on workload\n";
    std::cout << "if (problem.is_compute_bound())\n";
    std::cout << "    compute_disp.run(...);\n";
    std::cout << "else\n";
    std::cout << "    memory_disp.run(...);\n";
    print_separator();

    return all_passed ? 0 : 1;
}
