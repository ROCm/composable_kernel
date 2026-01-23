// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Correctness test with real GPU kernel
 * Validates GPU results against CPU reference implementation
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <hip/hip_runtime.h>

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

// Kernel header included via -include compiler flag

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using Priority = ck_tile::dispatcher::Registry::Priority;

#define HIP_CHECK(call)                                                   \
    {                                                                     \
        hipError_t err = call;                                            \
        if(err != hipSuccess)                                             \
        {                                                                 \
            std::cerr << "HIP Error: " << hipGetErrorString(err) << "\n"; \
            exit(1);                                                      \
        }                                                                 \
    }

// CPU reference GEMM
// A: RowMajor (M x K) - A[m,k] = A[m*K + k]
// B: ColumnMajor (K x N) - B[k,n] = B[k + n*K]
// C: RowMajor (M x N) - C[m,n] = C[m*N + n]
template <typename T>
void cpu_gemm(
    const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int M, int N, int K)
{
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < N; n++)
        {
            float acc = 0.0f;
            for(int k = 0; k < K; k++)
            {
                // A is row-major: A[m,k] = A[m*K + k]
                // B is column-major: B[k,n] = B[k + n*K]
                acc += float(A[m * K + k]) * float(B[k + n * K]);
            }
            C[m * N + n] = T(acc);
        }
    }
}

int main()
{
    std::cout << "=======================================\n";
    std::cout << "Correctness Test - Real GPU Kernel\n";
    std::cout << "=======================================\n\n";

    std::cout << "Kernel: " << KERNEL_NAME << "\n\n";

    // Register kernel
    KernelKey key;
    key.signature.dtype_a             = DataType::FP16;
    key.signature.dtype_b             = DataType::FP16;
    key.signature.dtype_c             = DataType::FP16;
    key.signature.dtype_acc           = DataType::FP32;
    key.signature.layout_a            = LayoutTag::RowMajor;
    key.signature.layout_b            = LayoutTag::ColMajor;
    key.signature.layout_c            = LayoutTag::RowMajor;
    key.signature.transpose_a         = false;
    key.signature.transpose_b         = false;
    key.signature.grouped             = false;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = false;

    key.algorithm.tile_shape      = {128, 128, 32};
    key.algorithm.wave_shape      = {2, 2, 1};
    key.algorithm.warp_tile_shape = {32, 32, 16};
    key.algorithm.pipeline        = Pipeline::CompV4;
    key.algorithm.scheduler       = Scheduler::Intrawave;
    key.algorithm.epilogue        = Epilogue::CShuffle;
    key.algorithm.block_size      = 256;
    key.algorithm.double_buffer   = true;
    key.algorithm.persistent      = false;
    key.algorithm.preshuffle      = false;
    key.algorithm.transpose_c     = false;
    key.algorithm.num_wave_groups = 1;
    key.gfx_arch                  = "gfx942";

    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Priority::High);

    Dispatcher dispatcher;

    // Test with random matrices
    const int M = 256;
    const int N = 256;
    const int K = 256;

    std::cout << "Test configuration:\n";
    std::cout << "  Problem: M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "  Method: Random matrices vs CPU reference\n\n";

    // Random number generation
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<ADataType> A_host(M * K);
    std::vector<BDataType> B_host(K * N);
    std::vector<CDataType> C_gpu(M * N);
    std::vector<CDataType> C_cpu(M * N);

    // Initialize with random values
    std::cout << "Initializing random matrices...\n";
    for(int i = 0; i < M * K; i++)
    {
        A_host[i] = ADataType(dist(rng));
    }
    for(int i = 0; i < K * N; i++)
    {
        B_host[i] = BDataType(dist(rng));
    }

    // GPU execution
    std::cout << "Executing on GPU...\n";

    ADataType *A_dev, *B_dev;
    CDataType* C_dev;

    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(CDataType)));

    HIP_CHECK(hipMemcpy(A_dev, A_host.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B_host.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(CDataType)));

    Problem problem(M, N, K);
    float gpu_time = dispatcher.run(A_dev, B_dev, C_dev, problem);

    HIP_CHECK(hipMemcpy(C_gpu.data(), C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));

    std::cout << "OK GPU execution complete: " << gpu_time << " ms\n";

    double flops  = 2.0 * M * N * K;
    double tflops = (flops / (gpu_time * 1e-3)) / 1e12;
    std::cout << "OK GPU performance: " << tflops << " TFLOPS\n\n";

    // CPU reference
    std::cout << "Computing CPU reference...\n";
    cpu_gemm(A_host, B_host, C_cpu, M, N, K);
    std::cout << "OK CPU reference complete\n\n";

    // Validation
    std::cout << "Validating results...\n";

    int num_correct       = 0;
    float max_rel_error   = 0.0f;
    float max_abs_error   = 0.0f;
    const float tolerance = 0.02f; // 2% for FP16

    for(int i = 0; i < M * N; i++)
    {
        float gpu_val = float(C_gpu[i]);
        float cpu_val = float(C_cpu[i]);

        float abs_error = std::abs(gpu_val - cpu_val);
        float rel_error = abs_error / (std::abs(cpu_val) + 1e-5f);

        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);

        if(rel_error < tolerance)
        {
            num_correct++;
        }
    }

    float accuracy = 100.0f * num_correct / (M * N);

    std::cout << "\nValidation Results:\n";
    std::cout << "  Correct elements: " << num_correct << "/" << M * N << "\n";
    std::cout << "  Accuracy: " << accuracy << "%\n";
    std::cout << "  Max absolute error: " << max_abs_error << "\n";
    std::cout << "  Max relative error: " << max_rel_error << "\n";
    std::cout << "  Tolerance: " << tolerance << " (2%)\n\n";

    // Show sample comparisons
    std::cout << "Sample results (first 5 elements):\n";
    std::cout << "  Index | GPU Result | CPU Result | Error\n";
    std::cout << "  ------|------------|------------|-------\n";

    for(int i = 0; i < 5; i++)
    {
        float gpu_val = float(C_gpu[i]);
        float cpu_val = float(C_cpu[i]);
        float error   = std::abs(gpu_val - cpu_val);
        printf("  %-5d | %10.4f | %10.4f | %.4f\n", i, gpu_val, cpu_val, error);
    }
    std::cout << "\n";

    // Cleanup
    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));

    if(accuracy > 99.0f)
    {
        std::cout << "[OK] CORRECTNESS TEST PASSED\n";
        std::cout << "   GPU results match CPU reference within tolerance\n";
        return 0;
    }
    else
    {
        std::cout << "[FAIL] CORRECTNESS TEST FAILED\n";
        std::cout << "   Accuracy too low: " << accuracy << "%\n";
        return 1;
    }
}
