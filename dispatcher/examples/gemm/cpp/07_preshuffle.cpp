// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 07: Preshuffle GEMM for Inference
 *
 * Demonstrates weight matrix preshuffling for optimized inference workloads.
 * Uses the dispatcher pattern for kernel selection.
 *
 * Build: cd dispatcher/build && cmake .. && make gemm_07_preshuffle
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// KERNEL SET: Standard GEMM kernels
// =============================================================================

DECL_KERNEL_SET(preshuffle_kernels,
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
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 07: Preshuffle GEMM", "Standard GEMM with dispatcher");
    args.add_option("--M", "2048", "Matrix M dimension");
    args.add_option("--N", "2048", "Matrix N dimension");
    args.add_option("--K", "1024", "Matrix K dimension");
    args.add_option("--arch", "gfx942", "GPU architecture");

    if(!args.parse(argc, argv))
        return 0;

    print_header("Example 07: Preshuffle GEMM for Inference");

    std::string gfx_arch = args.get("--arch", "gfx942");

    std::cout << "\nNote: This example demonstrates standard GEMM using the dispatcher.\n";
    std::cout << "      Preshuffle kernel support is planned for future releases.\n";

    // =========================================================================
    // Setup
    // =========================================================================
    std::cout << "\nSetup:\n";
    Registry registry;
    registry.set_name("inference_registry");

    generated::register_07_preshuffle_kernels(registry, gfx_arch);
    Dispatcher dispatcher(&registry);

    std::cout << "  Kernels: " << registry.size() << " registered\n";

    // =========================================================================
    // Prepare data
    // =========================================================================
    using DataType = ck_tile::fp16_t;

    const int M = args.get_int("--M", 2048);
    const int N = args.get_int("--N", 2048);
    const int K = args.get_int("--K", 1024);
    Problem problem(M, N, K);

    std::cout << "  Problem Size: " << M << " x " << N << " x " << K << "\n";

    std::vector<DataType> a_host(M * K, DataType(1.0f));
    std::vector<DataType> b_host(K * N, DataType(1.0f));

    GpuBuffer<DataType> a_dev(M * K);
    GpuBuffer<DataType> b_dev(K * N);
    GpuBuffer<DataType> c_dev(M * N);

    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());
    c_dev.zero();

    // =========================================================================
    // Run GEMM
    // =========================================================================
    std::cout << "\nRunning GEMM...\n";

    auto selected = dispatcher.select_kernel(problem);
    if(selected)
    {
        std::cout << "  Selected: " << selected->get_name() << "\n";
    }

    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);

    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << calculate_tflops(M, N, K, time_ms) << "\n";

    // =========================================================================
    // Verify ALL elements
    // =========================================================================
    std::vector<DataType> c_host(M * N);
    c_dev.copy_to_host(c_host.data());

    float expected  = static_cast<float>(K);
    int num_errors  = 0;
    float max_error = 0.0f;

    for(int i = 0; i < M * N; ++i)
    {
        float actual = static_cast<float>(c_host[i]);
        float error  = std::abs(actual - expected);
        max_error    = std::max(max_error, error);
        if(error > 0.01f * expected + 1.0f)
            ++num_errors;
    }

    bool passed = (num_errors == 0);

    print_separator();
    std::cout << "Verification: " << (M * N) << " elements checked\n";
    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  Errors: " << num_errors << "\n";
    std::cout << "Result: " << (passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    // =========================================================================
    // Inference pattern demo
    // =========================================================================
    if(passed)
    {
        std::cout << "\nInference Pattern (multiple calls with same weights):\n";
        print_separator();

        for(int i = 0; i < 3; i++)
        {
            c_dev.zero();
            float iter_time =
                dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
            std::cout << "  Inference " << (i + 1) << ": " << std::fixed << std::setprecision(4)
                      << iter_time << " ms\n";
        }
        print_separator();
    }

    return passed ? 0 : 1;
}
