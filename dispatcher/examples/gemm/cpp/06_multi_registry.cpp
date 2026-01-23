// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 06: Multiple Registries and Multiple Kernel Sets
 *
 * Demonstrates:
 * - Multiple DECL_KERNEL_SET declarations (each with multiple kernels)
 * - Separate Registry instances for different workload types
 * - Independent Dispatchers that select from their respective registries
 *
 * Registration patterns:
 * - REGISTER_GENERATED_KERNELS(registry, arch)         -> all kernels to one registry
 * - REGISTER_KERNEL_SET("set_name", registry, arch)    -> specific set by name
 * - generated::get_kernel_set_names()                  -> list available set names
 *
 * Build: cd dispatcher/build && cmake .. && make gemm_06_multi_registry
 * Usage: ./gemm_06_multi_registry [--list] [--help]
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
// KERNEL SETS: Multiple sets with multiple kernels each
// =============================================================================

// Compute-bound kernel set: Large tiles for high arithmetic intensity
// Max tile with 32x32 warp is 128x128 (16 warps = 1024 threads)
DECL_KERNEL_SET(compute_bound_set,
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(128, 128, 64) // Large tile, max for 32x32 warp
                         .wave(2, 2, 1)
                         .warp(32, 32, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx942")
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 128, 32) // Same tile, different K for variety
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942"));

// Memory-bound kernel set: Smaller tiles for better cache efficiency
DECL_KERNEL_SET(memory_bound_set,
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(64, 64, 32)
                         .wave(2, 2, 1)
                         .warp(32, 32, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx942")
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 64, 32)
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942"));

// Latency-optimized: Minimal overhead tiles
DECL_KERNEL_SET(latency_set,
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(64, 64, 64)
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
    ExampleArgs args("Example 06: Multiple Registries",
                     "Separate registries for different workload types");
    args.add_flag("--list", "List all declared kernel sets");
    args.add_option("--arch", "gfx942", "GPU architecture");

    if(!args.parse(argc, argv))
        return 0;

    print_header("Example 06: Multiple Registries & Kernel Sets");

    std::string gfx_arch = args.get("--arch", "gfx942");

    // =========================================================================
    // Step 1: Show declared kernel sets (from DECL_KERNEL_SET macros)
    // =========================================================================
    std::cout << "\nStep 1: Declared Kernel Sets\n";
    std::cout << "-----------------------------\n";
    KernelSetRegistry::instance().print();

    if(args.has("--list"))
    {
        // Print detailed info
        for(const auto& name : KernelSetRegistry::instance().names())
        {
            const auto& set = KernelSetRegistry::instance().get(name);
            std::cout << "\n  " << name << ":\n";
            for(const auto& decl : set.declarations())
            {
                std::cout << "    - " << decl.name() << " (tile=" << decl.algorithm.tile_m_ << "x"
                          << decl.algorithm.tile_n_ << "x" << decl.algorithm.tile_k_ << ")\n";
            }
        }
        return 0;
    }

    // =========================================================================
    // Step 2: Create registries and demonstrate MERGING
    // =========================================================================
    std::cout << "\nStep 2: Create and Merge Registries\n";
    std::cout << "------------------------------------\n";

    // Create individual registries first
    Registry compute_registry;
    Registry latency_registry;
    Registry memory_registry;

    compute_registry.set_name("compute_bound");
    latency_registry.set_name("latency_optimized");
    memory_registry.set_name("memory_bound");

    // Register kernels to individual registries using set names (no hardcoding)
    REGISTER_KERNEL_SET("compute_bound_set", compute_registry, gfx_arch);
    REGISTER_KERNEL_SET("latency_set", latency_registry, gfx_arch);
    REGISTER_KERNEL_SET("memory_bound_set", memory_registry, gfx_arch);

    std::cout << "  Individual registries:\n";
    std::cout << "    compute_bound: " << compute_registry.size() << " kernel(s)\n";
    std::cout << "    latency_optimized: " << latency_registry.size() << " kernel(s)\n";
    std::cout << "    memory_bound: " << memory_registry.size() << " kernel(s)\n";

    // MERGE compute + latency into a combined registry
    Registry combined_registry;
    combined_registry.set_name("compute_latency_combined");

    // Register both sets into combined registry
    REGISTER_KERNEL_SET("compute_bound_set", combined_registry, gfx_arch);
    REGISTER_KERNEL_SET("latency_set", combined_registry, gfx_arch);

    std::cout << "\n  After merging compute + latency:\n";
    std::cout << "    combined: " << combined_registry.size() << " kernel(s)\n";
    std::cout << "    memory (separate): " << memory_registry.size() << " kernel(s)\n";

    // =========================================================================
    // Step 3: Create dispatchers - one merged, one separate
    // =========================================================================
    std::cout << "\nStep 3: Create Dispatchers\n";
    std::cout << "--------------------------\n";

    Dispatcher combined_dispatcher(&combined_registry); // compute + latency merged
    Dispatcher memory_dispatcher(&memory_registry);     // memory separate

    std::cout << "  combined_dispatcher: compute + latency kernels (" << combined_registry.size()
              << " kernels)\n";
    std::cout << "  memory_dispatcher: memory-bound kernels (" << memory_registry.size()
              << " kernels)\n";

    // =========================================================================
    // Step 4: Run with different dispatchers
    // =========================================================================
    std::cout << "\nStep 4: Run Workloads\n";
    print_separator();

    using DataType = ck_tile::fp16_t;

    struct WorkloadTest
    {
        const char* name;
        Dispatcher* dispatcher;
        int M, N, K;
    };

    std::vector<WorkloadTest> tests = {
        {"Compute-bound (combined)", &combined_dispatcher, 4096, 4096, 4096},
        {"Memory-bound (separate)", &memory_dispatcher, 1024, 1024, 1024},
        {"Latency-opt (combined)", &combined_dispatcher, 512, 512, 512},
    };

    bool all_passed = true;

    for(const auto& test : tests)
    {
        Problem problem(test.M, test.N, test.K);

        // Allocate and initialize
        GpuBuffer<DataType> a_dev(test.M * test.K);
        GpuBuffer<DataType> b_dev(test.K * test.N);
        GpuBuffer<DataType> c_dev(test.M * test.N);

        std::vector<DataType> a_host(test.M * test.K, DataType(1.0f));
        std::vector<DataType> b_host(test.K * test.N, DataType(1.0f));
        a_dev.copy_from_host(a_host.data());
        b_dev.copy_from_host(b_host.data());
        c_dev.zero();

        // Select kernel and run
        auto selected = test.dispatcher->select_kernel(problem);
        float time_ms =
            test.dispatcher->run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
        double tflops = calculate_tflops(test.M, test.N, test.K, time_ms);

        std::cout << test.name << " (" << test.M << "x" << test.N << "x" << test.K << "):\n";
        if(selected)
            std::cout << "  Selected: " << selected->get_name() << "\n";
        std::cout << "  Time:     " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
        std::cout << "  TFLOPS:   " << std::setprecision(2) << tflops << "\n";

        // Verify ALL elements
        std::vector<DataType> c_host(test.M * test.N);
        c_dev.copy_to_host(c_host.data());
        const float expected = static_cast<float>(test.K);

        int num_errors  = 0;
        float max_error = 0.0f;
        for(int i = 0; i < test.M * test.N; ++i)
        {
            float actual = static_cast<float>(c_host[i]);
            float error  = std::abs(actual - expected);
            max_error    = std::max(max_error, error);
            // Allow 1% relative tolerance for FP16 accumulation
            if(error > 0.01f * expected + 1.0f)
                ++num_errors;
        }

        bool test_passed = (num_errors == 0);
        std::cout << "  Verify:   " << (test.M * test.N) << " elements, errors=" << num_errors
                  << "\n";
        std::cout << "  Status:   " << (test_passed ? "PASS" : "FAIL") << "\n\n";

        if(!test_passed)
            all_passed = false;
    }

    // =========================================================================
    // Summary
    // =========================================================================
    print_separator();
    std::cout << "Multi-Registry Pattern Summary:\n";
    print_separator();
    std::cout << R"(
// 1. Declare multiple kernel sets
DECL_KERNEL_SET(compute_bound_set, .add(...));
DECL_KERNEL_SET(memory_bound_set, .add(...));
DECL_KERNEL_SET(latency_set, .add(...));

// 2. Create registries and register by set NAME (no hardcoding!)
Registry combined_reg, memory_reg;
REGISTER_KERNEL_SET("compute_bound_set", combined_reg, arch);  // Add compute
REGISTER_KERNEL_SET("latency_set", combined_reg, arch);        // Merge latency
REGISTER_KERNEL_SET("memory_bound_set", memory_reg, arch);     // Separate

// 3. Create dispatchers from merged/separate registries
Dispatcher combined_disp(&combined_reg);  // Has both compute + latency
Dispatcher memory_disp(&memory_reg);      // Has only memory-bound

// 4. Choose dispatcher based on workload
if (problem.is_memory_bound())
    memory_disp.run(...);
else
    combined_disp.run(...);  // Handles both compute & latency workloads
)";
    print_separator();
    std::cout << "Overall Status: " << (all_passed ? "ALL PASSED" : "SOME FAILED") << "\n";

    return all_passed ? 0 : 1;
}
