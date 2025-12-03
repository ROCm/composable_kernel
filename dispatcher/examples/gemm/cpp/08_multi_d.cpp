// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 08: Multi-D GEMM (Fused Operations)
 *
 * Demonstrates GEMM with additional D tensors for fused operations.
 * E = ElementWise(A * B, D0, D1, ...)
 *
 * For example with MultiDMultiply:
 *   E = (A @ B) * D0 * D1
 *
 * The D tensors have the same shape as the output (M x N) and are loaded
 * during the epilogue phase, enabling fusion without extra memory passes.
 *
 * Key concepts:
 *   - GemmKernelMultiD: Special kernel that handles D tensor loading
 *   - GemmMultiDHostArgs: Host args with D tensor pointers and strides
 *   - DsDataType/DsLayout: Tuples defining D tensor types and layouts
 *   - ElementWiseFn: Fused operation (MultiDAdd, MultiDMultiply, Relu, etc.)
 *
 * This example uses a generated kernel via -include, like other examples.
 *
 * Build:
 *   cmake -DBUILD_DISPATCHER_EXAMPLES=ON ..
 *   make gemm_08_multi_d
 *
 * Complexity: ★★★★☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// Types from generated kernel (via -include)
// =============================================================================
// The generated kernel provides:
//   - SelectedKernel: The kernel struct
//   - ADataType, BDataType, CDataType: Data types
//   - NumDTensor: Number of D tensors
//   - GemmMultiDArgs: Host args type for Multi-D

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 08: Multi-D GEMM", "GEMM with fused D tensor operations");
    args.add_option("--M", "512", "Matrix M dimension");
    args.add_option("--N", "512", "Matrix N dimension");
    args.add_option("--K", "256", "Matrix K dimension");
    args.add_option("--warmup", "5", "Warmup iterations");
    args.add_option("--repeat", "20", "Benchmark iterations");
    args.add_flag("--verify", "Run CPU verification");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "\n======================================================================\n";
    std::cout << "Example 08: Multi-D GEMM (Fused Operations)\n";
    std::cout << "======================================================================\n";

    const int M       = args.get_int("--M", 512);
    const int N       = args.get_int("--N", 512);
    const int K       = args.get_int("--K", 256);
    const int warmup  = args.get_int("--warmup", 5);
    const int repeat  = args.get_int("--repeat", 20);
    const bool verify = args.has("--verify");

    std::cout << "\nMulti-D GEMM Configuration:\n";
    std::cout << "  Kernel:    " << KERNEL_NAME << "\n";
    std::cout << "  Operation: E = ElementWise(A @ B, D0, D1)\n";
    std::cout << "  Problem:   " << M << " x " << N << " x " << K << "\n";
    std::cout << "  D tensors: " << NumDTensor << " (each " << M << " x " << N << ")\n";
    std::cout << "\n";

    // =========================================================================
    // Setup tensors
    // =========================================================================
    std::cout << "Step 1: Initialize Tensors\n";
    std::cout << "--------------------------\n";

    // Host tensors
    HostTensor<ADataType> a_host({M, K});
    HostTensor<BDataType> b_host({K, N});
    HostTensor<CDataType> d0_host({M, N});
    HostTensor<CDataType> d1_host({M, N});
    HostTensor<CDataType> e_host({M, N});

    // Initialize with random values
    FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_host);
    FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_host);
    FillUniformDistribution<CDataType>{0.5f, 1.5f}(d0_host); // Positive for multiplication
    FillUniformDistribution<CDataType>{0.5f, 1.5f}(d1_host);

    std::cout << "  A:  " << M << " x " << K << " (fp16)\n";
    std::cout << "  B:  " << K << " x " << N << " (fp16)\n";
    std::cout << "  D0: " << M << " x " << N << " (fp16)\n";
    std::cout << "  D1: " << M << " x " << N << " (fp16)\n";
    std::cout << "  E:  " << M << " x " << N << " (fp16, output)\n\n";

    // Device memory
    DeviceMem a_dev(a_host.get_element_space_size_in_bytes());
    DeviceMem b_dev(b_host.get_element_space_size_in_bytes());
    DeviceMem d0_dev(d0_host.get_element_space_size_in_bytes());
    DeviceMem d1_dev(d1_host.get_element_space_size_in_bytes());
    DeviceMem e_dev(e_host.get_element_space_size_in_bytes());

    a_dev.ToDevice(a_host.data());
    b_dev.ToDevice(b_host.data());
    d0_dev.ToDevice(d0_host.data());
    d1_dev.ToDevice(d1_host.data());
    e_dev.SetZero();

    // =========================================================================
    // Setup kernel args
    // =========================================================================
    std::cout << "Step 2: Create GemmMultiDHostArgs\n";
    std::cout << "---------------------------------\n";

    // Strides (row-major for A, E, D; column-major for B)
    const index_t stride_A  = K; // Row-major: stride = K
    const index_t stride_B  = K; // Col-major: stride = K (leading dimension)
    const index_t stride_D0 = N; // Row-major
    const index_t stride_D1 = N; // Row-major
    const index_t stride_E  = N; // Row-major

    // D tensor pointers and strides as arrays
    std::array<const void*, NumDTensor> ds_ptrs = {d0_dev.GetDeviceBuffer(),
                                                   d1_dev.GetDeviceBuffer()};
    std::array<index_t, NumDTensor> ds_strides  = {stride_D0, stride_D1};

    GemmMultiDArgs kernel_args{a_dev.GetDeviceBuffer(),
                               b_dev.GetDeviceBuffer(),
                               ds_ptrs,
                               e_dev.GetDeviceBuffer(),
                               1, // k_batch (must be 1 for Multi-D)
                               M,
                               N,
                               K,
                               stride_A,
                               stride_B,
                               ds_strides,
                               stride_E};

    std::cout << "  D tensor pointers: " << ds_ptrs.size() << "\n";
    std::cout << "  D strides: [" << stride_D0 << ", " << stride_D1 << "]\n\n";

    // =========================================================================
    // Run kernel
    // =========================================================================
    std::cout << "Step 3: GPU Execution\n";
    std::cout << "---------------------\n";

    stream_config stream_cfg{nullptr, true, 0, warmup, repeat};

    float time_ms = SelectedKernel::launch(kernel_args, stream_cfg);

    double flops  = 2.0 * M * N * K + 2.0 * M * N * NumDTensor; // GEMM + element-wise ops
    double tflops = (flops / (time_ms / 1000.0)) / 1e12;

    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n\n";

    // =========================================================================
    // Verification
    // =========================================================================
    if(verify)
    {
        std::cout << "Step 4: CPU Verification\n";
        std::cout << "------------------------\n";

        // CPU reference: E = (A @ B) * D0 * D1 (for MultiDMultiply)
        HostTensor<CDataType> e_ref({M, N});

        // Compute GEMM: C = A @ B, then apply element-wise
        // Note: B is column-major, so b(k, n) accesses element at column n, row k
        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                float acc = 0.0f;
                for(int k = 0; k < K; ++k)
                {
                    // B is column-major: b[n * K + k]
                    acc += type_convert<float>(a_host(m, k)) *
                           type_convert<float>(b_host.data()[n * K + k]);
                }
                // Apply element-wise: E = C * D0 * D1
                float d0    = type_convert<float>(d0_host(m, n));
                float d1    = type_convert<float>(d1_host(m, n));
                e_ref(m, n) = type_convert<CDataType>(acc * d0 * d1);
            }
        }

        // Copy result back
        e_dev.FromDevice(e_host.data());

        // Compare
        bool pass = check_err(e_host, e_ref, "Multi-D GEMM verification", 0.05f, 0.05f);

        std::cout << "  Status: " << (pass ? "PASS" : "FAIL") << "\n\n";
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "======================================================================\n";
    std::cout << "Multi-D GEMM Pattern:\n";
    std::cout << "  1. D tensors loaded during epilogue (fused)\n";
    std::cout << "  2. Zero extra memory passes for element-wise ops\n";
    std::cout << "  3. Supports: MultiDAdd, MultiDMultiply, Relu, Gelu, etc.\n";
    std::cout << "  4. Use cases: Transformers, MLPs, Conv layers\n";
    std::cout << "======================================================================\n";

    return 0;
}
