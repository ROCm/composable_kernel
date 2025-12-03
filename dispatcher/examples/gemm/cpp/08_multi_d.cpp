// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 08: Multi-D GEMM (Fused Operations)
 *
 * Demonstrates GEMM with additional D tensors for fused operations.
 * C = A * B + D0 + D1 + ...
 *
 * Build:
 *   python3 scripts/compile_gemm_examples.py examples/cpp/08_multi_d.cpp
 *
 * Complexity: ★★★☆☆
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
// KERNEL SET: Multi-D kernels with fused elementwise
// =============================================================================

DECL_KERNEL_SET(
    multi_d,
    .add(Signature().dtype("fp16").layout("rcr").elementwise("MultiDAdd", 1), // 1 D tensor
         Algorithm().tile(128, 128, 32))
        .add(Signature().dtype("fp16").layout("rcr").elementwise("MultiDAdd", 2), // 2 D tensors
             Algorithm().tile(128, 128, 32)));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 08: Multi-D GEMM", "GEMM with fused D tensor operations");
    args.add_option("--M", "1024", "Matrix M dimension");
    args.add_option("--N", "1024", "Matrix N dimension");
    args.add_option("--K", "512", "Matrix K dimension");
    args.add_flag("--list", "List all kernel sets");

    if(!args.parse(argc, argv))
        return 0;

    print_header("Example 08: Multi-D GEMM (Fused Operations)");

    if(args.has("--list"))
    {
        std::cout << "\nDeclared Kernel Sets:\n";
        KernelSetRegistry::instance().print();
        return 0;
    }

    std::cout << "\nMulti-D GEMM supports:\n";
    std::cout << "  - C = A * B + D0 (bias add)\n";
    std::cout << "  - C = A * B + D0 + D1 (multiple additions)\n";
    std::cout << "  - C = ReLU(A * B + D0) (fused activation)\n";

    // =========================================================================
    // Setup
    // =========================================================================
    std::cout << "\nSetup:\n";
    Registry registry;
    registry.set_name("multi_d_registry");

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
    // Run GEMM (standard, without D tensors for this demo)
    // =========================================================================
    const int M = args.get_int("--M", 1024);
    const int N = args.get_int("--N", 1024);
    const int K = args.get_int("--K", 512);
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
    // Use 1% relative tolerance for FP16 accumulation over K elements
    bool passed = std::abs(actual - expected) < (0.01f * expected + 1.0f);

    print_separator();
    std::cout << "Result: C[0,0] = " << actual << " (expected " << expected << ")\n";
    std::cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n";
    print_separator();

    std::cout << "\nNote: This example uses standard GEMM.\n";
    std::cout << "For Multi-D, use dispatcher.run_with_d(...) with D tensor pointers.\n";

    return passed ? 0 : 1;
}
