// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// Real kernel test: Dispatcher with actual CK Tile kernels on GPU
// This test uses automatically generated kernels from unified_gemm_codegen.py

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <hip/hip_runtime.h>

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"

// Include auto-generated dispatcher wrappers
#include "dispatcher_wrappers/register_all_kernels.hpp"

using namespace ck_tile::dispatcher;
using ck_tile::dispatcher::Registry;
using ck_tile::dispatcher::Dispatcher;
using ck_tile::dispatcher::Problem;
using Priority = ck_tile::dispatcher::Registry::Priority;

#define HIP_CHECK(call) { \
    hipError_t err = call; \
    if(err != hipSuccess) { \
        std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << hipGetErrorString(err) << "\n"; \
        exit(1); \
    } \
}

// Reference CPU GEMM for validation
template<typename T>
void reference_gemm(
    const std::vector<T>& A,
    const std::vector<T>& B,
    std::vector<T>& C,
    int M, int N, int K)
{
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            float acc = 0.0f;
            for(int k = 0; k < K; k++) {
                acc += float(A[m * K + k]) * float(B[k * N + n]);
            }
            C[m * N + n] = T(acc);
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "=======================================\n";
    std::cout << "Real Kernel Dispatcher Test\n";
    std::cout << "=======================================\n\n";
    
    // Problem sizes (must be multiples of tile size for this kernel)
    const int M = 256;
    const int N = 256;
    const int K = 256;
    
    std::cout << "Problem: M=" << M << " N=" << N << " K=" << K << "\n\n";
    
    // Step 1: Register all auto-generated kernels
    Registry::instance().clear();
    register_all_tile_gemm_kernels(942, Priority::High);
    
    std::size_t kernel_count = get_tile_gemm_kernel_count();
    std::cout << "OK Registered " << kernel_count << " CK Tile kernels\n";
    
    // Step 2: Create dispatcher and problem
    Dispatcher dispatcher;
    Problem problem(M, N, K);
    
    // Step 3: Select kernel (dispatcher will choose best match)
    auto selected = dispatcher.select_kernel(problem);
    if (!selected) {
        std::cerr << "[FAIL] Failed to select kernel\n";
        return 1;
    }
    
    std::cout << "OK Selected kernel: " << selected->get_name() << "\n\n";
    
    // Step 4: Prepare test data (using FP16)
    using DataType = ck_tile::fp16_t;
    
    std::cout << "Preparing test data...\n";
    
    std::vector<DataType> A_host(M * K);
    std::vector<DataType> B_host(K * N);
    std::vector<DataType> C_gpu_result(M * N);
    std::vector<DataType> C_cpu_reference(M * N);
    
    // Initialize with random values
    for(int i = 0; i < M * K; i++) {
        A_host[i] = DataType(float(rand() % 10) / 10.0f);
    }
    for(int i = 0; i < K * N; i++) {
        B_host[i] = DataType(float(rand() % 10) / 10.0f);
    }
    
    std::cout << "OK Initialized random input matrices\n";
    
    // Step 5: Allocate GPU memory
    DataType *A_dev, *B_dev;
    DataType *C_dev;
    
    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(DataType)));
    
    std::cout << "OK Allocated GPU memory\n";
    
    // Step 6: Copy data to GPU
    HIP_CHECK(hipMemcpy(A_dev, A_host.data(), M * K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B_host.data(), K * N * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(DataType)));
    
    std::cout << "OK Copied data to GPU\n\n";
    
    // Step 7: Execute GPU kernel via dispatcher
    std::cout << "Executing GPU kernel...\n";
    float gpu_time = dispatcher.run(A_dev, B_dev, C_dev, problem);
    
    std::cout << "OK GPU execution time: " << gpu_time << " ms\n";
    
    // Calculate performance
    double flops = 2.0 * M * N * K;  // MAD ops
    double tflops = (flops / (gpu_time * 1e-3)) / 1e12;
    std::cout << "OK GPU performance: " << tflops << " TFLOPS\n\n";
    
    // Step 8: Copy result back
    HIP_CHECK(hipMemcpy(C_gpu_result.data(), C_dev, M * N * sizeof(DataType), 
                        hipMemcpyDeviceToHost));
    
    std::cout << "OK Copied results back to host\n";
    
    // Step 11: Compute CPU reference
    std::cout << "Computing CPU reference...\n";
    reference_gemm(A_host, B_host, C_cpu_reference, M, N, K);
    std::cout << "OK CPU reference computed\n\n";
    
    // Step 12: Validate results
    std::cout << "Validating results...\n";
    
    int num_correct = 0;
    int num_total = M * N;
    float max_error = 0.0f;
    float tolerance = 0.01f;  // 1% tolerance for FP16
    
    for(int i = 0; i < num_total; i++) {
        float gpu_val = float(C_gpu_result[i]);
        float cpu_val = float(C_cpu_reference[i]);
        float error = std::abs(gpu_val - cpu_val) / (std::abs(cpu_val) + 1e-5f);
        
        max_error = std::max(max_error, error);
        
        if(error < tolerance) {
            num_correct++;
        }
    }
    
    float accuracy = 100.0f * num_correct / num_total;
    
    std::cout << "Results:\n";
    std::cout << "  Correct elements: " << num_correct << "/" << num_total << "\n";
    std::cout << "  Accuracy: " << accuracy << "%\n";
    std::cout << "  Max error: " << max_error << "\n\n";
    
    // Sample outputs
    std::cout << "Sample results (first 5 elements):\n";
    for(int i = 0; i < 5; i++) {
        std::cout << "  C[" << i << "]: GPU=" << float(C_gpu_result[i]) 
                  << " CPU=" << float(C_cpu_reference[i]) << "\n";
    }
    std::cout << "\n";
    
    // Step 13: Cleanup
    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));
    
    std::cout << "OK Cleaned up GPU memory\n\n";
    
    // Final result
    if(accuracy > 99.9f) {
        std::cout << "[OK] TEST PASSED - Dispatcher executed real kernel correctly!\n";
        return 0;
    } else {
        std::cout << "[FAIL] TEST FAILED - Accuracy too low: " << accuracy << "%\n";
        return 1;
    }
}

