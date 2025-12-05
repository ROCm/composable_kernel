// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 08: Multi-D GEMM (Fused Operations)
 *
 * Demonstrates GEMM with additional D tensors for fused operations.
 * E = ElementWise(A * B, D0, D1, ...)
 *
 * Note: Multi-D GEMM requires specialized kernel registration.
 * This example demonstrates a standard GEMM using the dispatcher pattern
 * as a placeholder until multi-D kernel registration is complete.
 *
 * Build: cd dispatcher/build && cmake .. && make gemm_08_multi_d
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
// KERNEL SET: Standard GEMM for now (Multi-D registration TBD)
// =============================================================================

DECL_KERNEL_SET(multi_d_kernels,
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
    ExampleArgs args("Example 08: Multi-D GEMM", "GEMM demonstration (Multi-D TBD)");
    args.add_option("--M", "512", "Matrix M dimension");
    args.add_option("--N", "512", "Matrix N dimension");
    args.add_option("--K", "256", "Matrix K dimension");
    args.add_option("--arch", "gfx942", "GPU architecture");
    args.add_flag("--verify", "Run verification");

    if(!args.parse(argc, argv))
        return 0;

    print_header("Example 08: Multi-D GEMM (Fused Operations)");

    std::string gfx_arch = args.get("--arch", "gfx942");
    const int M          = args.get_int("--M", 512);
    const int N          = args.get_int("--N", 512);
    const int K          = args.get_int("--K", 256);
    const bool verify    = args.has("--verify");

    std::cout << "\nNote: This example demonstrates standard GEMM using the dispatcher.\n";
    std::cout << "      Multi-D kernel (E = ElementWise(A @ B, D0, D1)) support planned.\n";

    std::cout << "\nConfiguration:\n";
    std::cout << "  Problem:   " << M << " x " << N << " x " << K << "\n";

    // =========================================================================
    // Setup Registry and Dispatcher
    // =========================================================================
    Registry registry;
    generated::register_08_multi_d_kernels(registry, gfx_arch);
    Dispatcher dispatcher(&registry);

    std::cout << "  Kernels:   " << registry.size() << " registered\n";

    // =========================================================================
    // Setup tensors
    // =========================================================================
    std::cout << "\nStep 1: Initialize Tensors\n";
    std::cout << "--------------------------\n";

    using DataType = ck_tile::fp16_t;

    std::vector<DataType> a_host(M * K, DataType(1.0f));
    std::vector<DataType> b_host(K * N, DataType(1.0f));

    GpuBuffer<DataType> a_dev(M * K);
    GpuBuffer<DataType> b_dev(K * N);
    GpuBuffer<DataType> c_dev(M * N);

    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());
    c_dev.zero();

    std::cout << "  A:  " << M << " x " << K << " (fp16)\n";
    std::cout << "  B:  " << K << " x " << N << " (fp16)\n";
    std::cout << "  C:  " << M << " x " << N << " (fp16, output)\n\n";

    // =========================================================================
    // Run kernel
    // =========================================================================
    std::cout << "Step 2: GPU Execution\n";
    std::cout << "---------------------\n";

    Problem problem(M, N, K);

    auto selected = dispatcher.select_kernel(problem);
    if(selected)
    {
        std::cout << "  Selected: " << selected->get_name() << "\n";
    }

    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);

    double flops  = 2.0 * M * N * K;
    double tflops = (flops / (time_ms / 1000.0)) / 1e12;

    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n\n";

    // =========================================================================
    // Verification
    // =========================================================================
    bool pass = true;

    if(verify)
    {
        std::cout << "Step 3: Verification\n";
        std::cout << "--------------------\n";

        std::vector<DataType> c_host(M * N);
        c_dev.copy_to_host(c_host.data());

        float expected = static_cast<float>(K);
        int num_errors = 0;

        for(int i = 0; i < M * N; ++i)
        {
            float actual = static_cast<float>(c_host[i]);
            if(std::abs(actual - expected) > 0.01f * expected + 1.0f)
                ++num_errors;
        }

        pass = (num_errors == 0);
        std::cout << "  Elements checked: " << (M * N) << "\n";
        std::cout << "  Errors: " << num_errors << "\n";
        std::cout << "  Status: " << (pass ? "PASS" : "FAIL") << "\n\n";
    }

    // =========================================================================
    // Summary
    // =========================================================================
    print_separator();
    std::cout << "Multi-D GEMM Pattern (planned):\n";
    std::cout << "  1. D tensors loaded during epilogue (fused)\n";
    std::cout << "  2. Zero extra memory passes for element-wise ops\n";
    std::cout << "  3. Supports: MultiDAdd, MultiDMultiply, etc.\n";
    std::cout << "  4. Use cases: Transformers, MLPs, Conv layers\n";
    print_separator();

    return pass ? 0 : 1;
}
