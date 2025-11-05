// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Test with KNOWN matrices to verify correctness
 * 
 * Tests:
 * 1. Identity matrix: I * I = I
 * 2. All ones: ones * ones = K * ones (each element = K)
 * 3. Simple pattern: Sequential values
 */

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;

#define HIP_CHECK(call) { \
    hipError_t err = call; \
    if(err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << "\n"; \
        exit(1); \
    } \
}

void test_all_ones(Dispatcher& dispatcher, int M, int N, int K)
{
    std::cout << "\n======================================================================\n";
    std::cout << "TEST 1: All Ones Matrix\n";
    std::cout << "======================================================================\n";
    std::cout << "A = all 1s (MxK), B = all 1s (KxN)\n";
    std::cout << "Expected: C[i,j] = K (sum of K products of 1*1)\n\n";
    
    // Allocate
    ADataType *a_dev, *b_dev;
    CDataType *c_dev;
    HIP_CHECK(hipMalloc(&a_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&b_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&c_dev, M * N * sizeof(CDataType)));
    
    // Initialize host data - all ones
    std::vector<ADataType> a_host(M * K, ADataType(1.0f));
    std::vector<BDataType> b_host(K * N, BDataType(1.0f));
    std::vector<CDataType> c_result(M * N);
    
    // Copy to GPU
    HIP_CHECK(hipMemcpy(a_dev, a_host.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_dev, b_host.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(c_dev, 0, M * N * sizeof(CDataType)));
    
    // Execute
    Problem problem(M, N, K);
    float time = dispatcher.run(a_dev, b_dev, c_dev, problem, nullptr);
    
    // Get result
    HIP_CHECK(hipMemcpy(c_result.data(), c_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));
    
    // Verify: Every element should be K
    float expected = static_cast<float>(K);
    int correct = 0;
    int shown = 0;
    
    std::cout << "GPU Results (showing first 10 + last 5):\n";
    for(int i = 0; i < M * N; i++) {
        float val = static_cast<float>(c_result[i]);
        float diff = std::abs(val - expected);
        
        if(diff < 0.1f) correct++;
        
        if(shown < 10 || i >= M*N - 5) {
            std::cout << "  C[" << i << "] = " << val << " (expected " << expected 
                      << ", diff=" << diff << (diff < 0.1f ? " [OK]" : " [FAIL]") << ")\n";
            shown++;
        }
    }
    
    std::cout << "\nResult: " << correct << "/" << M*N << " correct ("
              << (100.0f * correct / (M*N)) << "%)\n";
    
    if(correct == M * N) {
        std::cout << "[OK] TEST PASSED - All ones multiplication correct!\n";
    } else {
        std::cout << "[FAIL] TEST FAILED - Only " << (100.0f*correct/(M*N)) << "% correct\n";
    }
    
    HIP_CHECK(hipFree(a_dev));
    HIP_CHECK(hipFree(b_dev));
    HIP_CHECK(hipFree(c_dev));
}

void test_identity_matrix(Dispatcher& dispatcher, int N)
{
    std::cout << "\n======================================================================\n";
    std::cout << "TEST 2: Identity Matrix\n";
    std::cout << "======================================================================\n";
    std::cout << "A = I (identity), B = sequential values\n";
    std::cout << "Expected: C = B (identity property)\n\n";
    
    // For square matrices: A = I (NxN), B = sequential (NxN)
    int M = N, K = N;
    
    // Allocate
    ADataType *a_dev, *b_dev;
    CDataType *c_dev;
    HIP_CHECK(hipMalloc(&a_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&b_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&c_dev, M * N * sizeof(CDataType)));
    
    // Initialize: A = identity matrix
    std::vector<ADataType> a_host(M * K, ADataType(0.0f));
    for(int i = 0; i < N; i++) {
        a_host[i * K + i] = ADataType(1.0f);  // Diagonal = 1
    }
    
    // B = sequential values
    // Column-major storage: b[k,n] is stored at index [n * K + k]
    std::vector<BDataType> b_host(K * N);
    for(int k = 0; k < K; k++) {
        for(int n = 0; n < N; n++) {
            // Column-major: column n, row k → index = n * leading_dim + k = n * K + k
            b_host[n * K + k] = BDataType(k + n * K);
        }
    }
    
    std::vector<CDataType> c_result(M * N);
    
    // Copy to GPU
    HIP_CHECK(hipMemcpy(a_dev, a_host.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_dev, b_host.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(c_dev, 0, M * N * sizeof(CDataType)));
    
    // Execute
    Problem problem(M, N, K);
    dispatcher.run(a_dev, b_dev, c_dev, problem, nullptr);
    
    // Get result
    HIP_CHECK(hipMemcpy(c_result.data(), c_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));
    
    // Verify: C should equal B (since A is identity)
    int correct = 0;
    std::cout << "First 10 results (C should = B):\n";
    for(int i = 0; i < std::min(10, M*N); i++) {
        int m = i / N;  // Row index in C (row-major)
        int n = i % N;  // Column index in C
        // For identity: C[m,n] = sum_k I[m,k] * B[k,n] = I[m,m] * B[m,n] = B[m,n]
        // B is column-major stored: B[k=m, n] at index [n * K + m]
        float expected = static_cast<float>(b_host[n * K + m]);
        float actual = static_cast<float>(c_result[i]);
        float diff = std::abs(actual - expected);
        
        if(diff < 0.1f) correct++;
        
        std::cout << "  C[" << m << "," << n << "] = " << actual 
                  << " (expected " << expected 
                  << ", diff=" << diff << (diff < 0.1f ? " [OK]" : " [FAIL]") << ")\n";
    }
    
    std::cout << "\nChecking all " << M*N << " elements...\n";
    correct = 0;
    for(int i = 0; i < M * N; i++) {
        int m = i / N;
        int n = i % N;
        float expected = static_cast<float>(b_host[n * K + m]);
        float actual = static_cast<float>(c_result[i]);
        if(std::abs(actual - expected) < 0.1f) correct++;
    }
    
    std::cout << "Result: " << correct << "/" << M*N << " correct ("
              << (100.0f * correct / (M*N)) << "%)\n";
    
    if(correct == M * N) {
        std::cout << "[OK] TEST PASSED - Identity matrix multiplication correct!\n";
    } else {
        std::cout << "[FAIL] TEST FAILED\n";
    }
    
    HIP_CHECK(hipFree(a_dev));
    HIP_CHECK(hipFree(b_dev));
    HIP_CHECK(hipFree(c_dev));
}

int main(int argc, char** argv)
{
    std::cout << "======================================================================\n";
    std::cout << "CK Tile Dispatcher - Known Matrix Verification\n";
    std::cout << "======================================================================\n";
    
    // Setup dispatcher
    KernelKey key;
    key.signature.dtype_a = DataType::FP16;
    key.signature.dtype_b = DataType::FP16;
    key.signature.dtype_c = DataType::FP16;
    key.signature.dtype_acc = DataType::FP32;
    key.signature.layout_a = LayoutTag::RowMajor;
    key.signature.layout_b = LayoutTag::ColMajor;
    key.signature.layout_c = LayoutTag::RowMajor;
    key.signature.elementwise_op = "PassThrough";
    key.signature.split_k = 1;
    
    key.algorithm.tile_shape = {128, 128, 64};
    key.algorithm.wave_shape = {2, 2, 1};
    key.algorithm.warp_tile_shape = {32, 32, 16};
    key.algorithm.pipeline = Pipeline::CompV4;
    key.algorithm.scheduler = Scheduler::Intrawave;
    key.algorithm.epilogue = Epilogue::CShuffle;
    key.algorithm.block_size = 256;
    key.algorithm.double_buffer = true;
    key.gfx_arch = 942;
    
    auto kernel = create_generated_tile_kernel<
        SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
        key, std::string(KERNEL_NAME));
    
    Registry::instance().clear();
    Registry::instance().register_kernel(kernel);
    
    Dispatcher dispatcher;
    
    // Run tests with known matrices
    int test_size = 128;  // Small for manual verification
    if(argc >= 2) {
        test_size = std::stoi(argv[1]);
    }
    
    test_all_ones(dispatcher, test_size, test_size, test_size);
    test_identity_matrix(dispatcher, test_size);
    
    return 0;
}

