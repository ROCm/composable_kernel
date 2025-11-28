// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 04: GEMM Validation
 *
 * Validates GEMM output against CPU reference computation.
 *
 * Build:
 *   python3 scripts/build_with_kernels.py examples/cpp/04_validation.cpp
 *
 * Complexity: ★★☆☆☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL SET
// =============================================================================

DECL_KERNEL_SET(validation, .add("fp16", "rcr", 128, 128, 32));

// =============================================================================
// CPU Reference
// =============================================================================

void gemm_reference_rcr(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        int M,
                        int N,
                        int K)
{
    // C = A * B^T for RCR layout (B is column-major = B^T is row-major)
    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k)
            {
                // A is row-major: A[m,k] = A[m * K + k]
                // B is col-major: B[k,n] = B[n * K + k]
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main()
{
    print_header("Example 04: GEMM Validation");

    const int M = 256, N = 256, K = 128;
    const float tolerance = 1e-2f;

    std::cout << "\nConfiguration:\n";
    std::cout << "  Problem:   " << M << " x " << N << " x " << K << "\n";
    std::cout << "  Layout:    RCR (A=row, B=col, C=row)\n";
    std::cout << "  Tolerance: " << tolerance << "\n";

    // =========================================================================
    // Setup
    // =========================================================================
    Registry registry;
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

    // =========================================================================
    // Initialize with random data
    // =========================================================================
    std::cout << "\nGenerating random test data...\n";
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a_fp32(M * K), b_fp32(K * N), c_ref(M * N);
    std::vector<ADataType> a_fp16(M * K);
    std::vector<BDataType> b_fp16(K * N);

    for(int i = 0; i < M * K; ++i)
    {
        a_fp32[i] = dist(rng);
        a_fp16[i] = ADataType(a_fp32[i]);
    }
    for(int i = 0; i < K * N; ++i)
    {
        b_fp32[i] = dist(rng);
        b_fp16[i] = BDataType(b_fp32[i]);
    }

    // =========================================================================
    // Compute reference
    // =========================================================================
    std::cout << "Computing CPU reference...\n";
    gemm_reference_rcr(a_fp32, b_fp32, c_ref, M, N, K);

    // =========================================================================
    // Run GPU kernel
    // =========================================================================
    std::cout << "Running GPU kernel...\n";

    GpuBuffer<ADataType> a_dev(M * K);
    GpuBuffer<BDataType> b_dev(K * N);
    GpuBuffer<CDataType> c_dev(M * N);

    a_dev.copy_from_host(a_fp16.data());
    b_dev.copy_from_host(b_fp16.data());
    c_dev.zero();

    Problem problem(M, N, K);
    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);

    std::vector<CDataType> c_gpu(M * N);
    c_dev.copy_to_host(c_gpu.data());

    std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";

    // =========================================================================
    // Validate
    // =========================================================================
    std::cout << "\nValidating...\n";

    int errors         = 0;
    float max_diff     = 0.0f;
    float max_rel_diff = 0.0f;

    for(int i = 0; i < M * N; ++i)
    {
        float gpu_val  = static_cast<float>(c_gpu[i]);
        float ref_val  = c_ref[i];
        float diff     = std::abs(gpu_val - ref_val);
        float rel_diff = (ref_val != 0.0f) ? diff / std::abs(ref_val) : diff;

        max_diff     = std::max(max_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);

        if(rel_diff > tolerance)
        {
            if(errors < 5)
            {
                int m = i / N, n = i % N;
                std::cout << "  Mismatch at [" << m << "," << n << "]: " << "GPU=" << gpu_val
                          << " REF=" << ref_val << " diff=" << diff << "\n";
            }
            errors++;
        }
    }

    print_separator();
    std::cout << "Validation Results:\n";
    print_separator();
    std::cout << "  Max absolute diff: " << max_diff << "\n";
    std::cout << "  Max relative diff: " << max_rel_diff << "\n";
    std::cout << "  Errors: " << errors << " / " << (M * N) << "\n";
    std::cout << "  Status: " << (errors == 0 ? "PASS" : "FAIL") << "\n";
    print_separator();

    return errors == 0 ? 0 : 1;
}
