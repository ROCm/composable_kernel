// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Simple real kernel test using tile_engine style (single kernel with -include)
 * This follows the proven pattern from the examples
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <hip/hip_runtime.h>

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

// Kernel header will be included via -include compiler flag
// It defines: ADataType, BDataType, CDataType, AccDataType, SelectedKernel, KERNEL_NAME

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

// Reference CPU GEMM
template <typename T>
void reference_gemm(
    const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int M, int N, int K)
{
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < N; n++)
        {
            float acc = 0.0f;
            for(int k = 0; k < K; k++)
            {
                acc += float(A[m * K + k]) * float(B[k * N + n]);
            }
            C[m * N + n] = T(acc);
        }
    }
}

int main()
{
    std::cout << "=======================================\n";
    std::cout << "Simple Real Kernel Test\n";
    std::cout << "=======================================\n\n";

    // Test size (must be multiple of tile size)
    const int M = 256;
    const int N = 256;
    const int K = 256;

    std::cout << "Problem: M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "Kernel: " << KERNEL_NAME << "\n\n";

    // Create kernel key
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

    key.algorithm.tile_shape      = {128, 128, 64};
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

    // Create and register kernel
    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Priority::High);

    std::cout << "OK Registered kernel\n";

    // Create dispatcher
    Dispatcher dispatcher;
    Problem problem(M, N, K);

    auto selected = dispatcher.select_kernel(problem);
    if(!selected)
    {
        std::cerr << "[FAIL] Failed to select kernel\n";
        return 1;
    }
    std::cout << "OK Selected kernel: " << selected->get_name() << "\n\n";

    // Prepare data
    std::cout << "Preparing test data...\n";
    std::vector<ADataType> A_host(M * K);
    std::vector<BDataType> B_host(K * N);
    std::vector<CDataType> C_gpu(M * N);
    std::vector<CDataType> C_cpu(M * N);

    // Simple test: A=1, B=1, C should be K
    for(int i = 0; i < M * K; i++)
        A_host[i] = ADataType(1.0f);
    for(int i = 0; i < K * N; i++)
        B_host[i] = BDataType(1.0f);

    // Allocate GPU memory
    ADataType *A_dev, *B_dev;
    CDataType* C_dev;

    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(CDataType)));

    HIP_CHECK(hipMemcpy(A_dev, A_host.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B_host.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(CDataType)));

    std::cout << "OK Data ready on GPU\n\n";

    // Execute
    std::cout << "Executing GPU kernel...\n";
    float gpu_time = dispatcher.run(A_dev, B_dev, C_dev, problem);

    std::cout << "OK GPU time: " << gpu_time << " ms\n";

    double flops  = 2.0 * M * N * K;
    double tflops = (flops / (gpu_time * 1e-3)) / 1e12;
    std::cout << "OK Performance: " << tflops << " TFLOPS\n\n";

    // Copy result
    HIP_CHECK(hipMemcpy(C_gpu.data(), C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));

    // Validate
    std::cout << "Validating (expected: all elements = " << K << ")...\n";

    int correct = 0;
    for(int i = 0; i < M * N; i++)
    {
        float val = float(C_gpu[i]);
        if(std::abs(val - float(K)) < 1.0f)
        {
            correct++;
        }
    }

    float accuracy = 100.0f * correct / (M * N);
    std::cout << "Accuracy: " << accuracy << "% (" << correct << "/" << M * N << ")\n";

    // Show samples
    std::cout << "\nFirst 5 results:\n";
    for(int i = 0; i < 5; i++)
    {
        std::cout << "  C[" << i << "] = " << float(C_gpu[i]) << " (expected " << K << ")\n";
    }
    std::cout << "\n";

    // Cleanup
    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));

    if(accuracy > 99.0f)
    {
        std::cout << "[OK] TEST PASSED\n";
        return 0;
    }
    else
    {
        std::cout << "[FAIL] TEST FAILED\n";
        return 1;
    }
}
