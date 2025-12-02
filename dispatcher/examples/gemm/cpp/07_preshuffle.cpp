// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 07: Preshuffle GEMM for Inference
 *
 * Demonstrates weight matrix preshuffling for optimized inference workloads.
 *
 * Preshuffle transforms the B (weight) matrix layout on the HOST before
 * sending to GPU. This allows the kernel to use optimized memory access
 * patterns, reducing bank conflicts and improving throughput.
 *
 * Benefits:
 * - Weights are fixed during inference, so shuffle once, use many times
 * - Optimized warp-level memory access patterns
 * - Reduced LDS bank conflicts
 * - Best for large matrices (2048+)
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
#include <cstring>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL SET: Inference-optimized kernels
// =============================================================================

DECL_KERNEL_SET(inference_optimized,
                .add("fp16", "rcr", 128, 128, 32).add("fp16", "rcr", 256, 256, 64));

// =============================================================================
// PRESHUFFLE UTILITIES
// =============================================================================

/**
 * Preshuffle the B (weight) matrix for optimized GEMM inference.
 *
 * This transforms the B matrix layout to match the expected memory access
 * pattern for preshuffle-enabled kernels. The transformation reorders data
 * so that warp-level loads are coalesced.
 *
 * @param b_src     Source B matrix (K x N) in column-major layout
 * @param b_dst     Destination buffer for shuffled B (same size)
 * @param K         K dimension
 * @param N         N dimension
 * @param warp_n    Warp tile size in N dimension (e.g., 32)
 * @param warp_k    Warp tile size in K dimension (e.g., 16)
 */
template <typename T>
void preshuffle_weight_matrix(const T* b_src, T* b_dst, int K, int N, int warp_n, int warp_k)
{
    // GFX9 (CDNA) preshuffle pattern
    // Based on ck_tile/host/tensor_shuffle_utils.hpp shuffle_b<>()
    //
    // Original layout: B[k, n] with K rows, N cols (column-major for 'c' layout)
    // Shuffled layout: Reordered for warp-level coalesced access
    //
    // Transformation:
    //   Reshape (K, N) -> (N/warp_n, warp_n, K/warp_k, divisor, warp_k/divisor)
    //   Permute with {0, 2, 3, 1, 4}

    int divisor = (warp_n == 32) ? 2 : 4;

    int n_tiles = N / warp_n;
    int k_tiles = K / warp_k;
    int k_inner = warp_k / divisor;

    // Perform the shuffle transformation
    for(int nt = 0; nt < n_tiles; nt++)
    {
        for(int kt = 0; kt < k_tiles; kt++)
        {
            for(int d = 0; d < divisor; d++)
            {
                for(int ni = 0; ni < warp_n; ni++)
                {
                    for(int ki = 0; ki < k_inner; ki++)
                    {
                        // Source index: B[k, n] where k = kt*warp_k + d*k_inner + ki
                        //                              n = nt*warp_n + ni
                        int k_src   = kt * warp_k + d * k_inner + ki;
                        int n_src   = nt * warp_n + ni;
                        int src_idx = k_src * N + n_src; // Column-major

                        // Destination index after permute {0, 2, 3, 1, 4}
                        // Shape: (n_tiles, k_tiles, divisor, warp_n, k_inner)
                        int dst_idx = nt * (k_tiles * divisor * warp_n * k_inner) +
                                      kt * (divisor * warp_n * k_inner) + d * (warp_n * k_inner) +
                                      ni * k_inner + ki;

                        b_dst[dst_idx] = b_src[src_idx];
                    }
                }
            }
        }
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 07: Preshuffle GEMM", "Weight matrix preshuffling for inference");
    args.add_option("--M", "2048", "Matrix M dimension");
    args.add_option("--N", "2048", "Matrix N dimension");
    args.add_option("--K", "1024", "Matrix K dimension");
    args.add_flag("--preshuffle", "Enable preshuffle transformation");
    args.add_flag("--list", "List all kernel sets");

    if(!args.parse(argc, argv))
        return 0;

    print_header("Example 07: Preshuffle GEMM for Inference");

    if(args.has("--list"))
    {
        std::cout << "\nDeclared Kernel Sets:\n";
        KernelSetRegistry::instance().print();
        return 0;
    }

    bool do_preshuffle = args.has("--preshuffle");

    std::cout << "\nPreshuffle Mode: " << (do_preshuffle ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "\nPreshuffle Benefits:\n";
    std::cout << "  - Weight matrix is pre-transformed offline\n";
    std::cout << "  - Optimized warp-level memory access patterns\n";
    std::cout << "  - Shuffle once, reuse for many inference calls\n";

    // =========================================================================
    // Setup
    // =========================================================================
    std::cout << "\nSetup:\n";
    Registry registry;
    registry.set_name("inference_registry");

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
    std::cout << "  Warp Tile: " << SelectedKernel::WarpTileM << "x" << SelectedKernel::WarpTileN
              << "x" << SelectedKernel::WarpTileK << "\n";

    // =========================================================================
    // Prepare data
    // =========================================================================
    const int M = args.get_int("--M", 2048);
    const int N = args.get_int("--N", 2048);
    const int K = args.get_int("--K", 1024);
    Problem problem(M, N, K);

    std::cout << "\nProblem Size: " << M << " x " << N << " x " << K << "\n";

    // Allocate host buffers
    std::vector<ADataType> a_host(M * K, ADataType(1.0f));
    std::vector<BDataType> b_host(K * N, BDataType(1.0f));
    std::vector<BDataType> b_shuffled(K * N);

    // Apply preshuffle transformation if enabled
    if(do_preshuffle)
    {
        std::cout << "\nPreshuffling weight matrix B...\n";
        preshuffle_weight_matrix<BDataType>(b_host.data(),
                                            b_shuffled.data(),
                                            K,
                                            N,
                                            SelectedKernel::WarpTileN,
                                            SelectedKernel::WarpTileK);
        std::cout << "  Preshuffle complete.\n";
    }

    // Allocate GPU buffers
    GpuBuffer<ADataType> a_dev(M * K);
    GpuBuffer<BDataType> b_dev(K * N);
    GpuBuffer<CDataType> c_dev(M * N);

    // Copy to GPU
    a_dev.copy_from_host(a_host.data());
    if(do_preshuffle)
    {
        b_dev.copy_from_host(b_shuffled.data());
    }
    else
    {
        b_dev.copy_from_host(b_host.data());
    }
    c_dev.zero();

    // =========================================================================
    // Run GEMM
    // =========================================================================
    std::cout << "\nRunning GEMM...\n";
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

    // =========================================================================
    // Inference pattern demonstration
    // =========================================================================
    if(do_preshuffle && passed)
    {
        std::cout << "\nInference Pattern (shuffle once, use many times):\n";
        print_separator();

        // Run multiple inference calls with same shuffled weights
        for(int i = 0; i < 3; i++)
        {
            // In real inference, A would be different activations
            c_dev.zero();
            float iter_time =
                dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
            std::cout << "  Inference " << (i + 1) << ": " << std::fixed << std::setprecision(4)
                      << iter_time << " ms\n";
        }
        print_separator();
    }

    std::cout << "\nUsage:\n";
    std::cout << "  ./07_preshuffle                  # Standard GEMM\n";
    std::cout << "  ./07_preshuffle --preshuffle     # With weight preshuffling\n";

    return passed ? 0 : 1;
}
