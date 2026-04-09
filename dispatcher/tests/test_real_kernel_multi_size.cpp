// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Multi-size real kernel test: Test multiple problem sizes with real GPU kernel
 */

#include <iostream>
#include <vector>
#include <cmath>
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

struct TestResult
{
    int M, N, K;
    float time_ms;
    double tflops;
    int correct;
    int total;
    bool passed;
};

TestResult run_test(Dispatcher& dispatcher, int M, int N, int K)
{
    TestResult result = {M, N, K, 0.0f, 0.0, 0, M * N, false};

    // Allocate and prepare data
    std::vector<ADataType> A_host(M * K);
    std::vector<BDataType> B_host(K * N);
    std::vector<CDataType> C_gpu(M * N);

    // Initialize: A=1, B=1, expected C=K
    for(int i = 0; i < M * K; i++)
        A_host[i] = ADataType(1.0f);
    for(int i = 0; i < K * N; i++)
        B_host[i] = BDataType(1.0f);

    ADataType *A_dev, *B_dev;
    CDataType* C_dev;

    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(CDataType)));

    HIP_CHECK(hipMemcpy(A_dev, A_host.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B_host.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(CDataType)));

    // Execute
    Problem problem(M, N, K);
    result.time_ms = dispatcher.run(A_dev, B_dev, C_dev, problem);

    // Calculate performance
    double flops  = 2.0 * M * N * K;
    result.tflops = (flops / (result.time_ms * 1e-3)) / 1e12;

    // Copy result and validate
    HIP_CHECK(hipMemcpy(C_gpu.data(), C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));

    for(int i = 0; i < M * N; i++)
    {
        if(std::abs(float(C_gpu[i]) - float(K)) < 1.0f)
        {
            result.correct++;
        }
    }

    result.passed = (result.correct == result.total);

    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));

    return result;
}

int main()
{
    std::cout << "=======================================\n";
    std::cout << "Multi-Size Real Kernel Test\n";
    std::cout << "=======================================\n\n";

    std::cout << "Using kernel: " << KERNEL_NAME << "\n\n";

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

    std::cout << "Running tests on multiple problem sizes...\n";
    std::cout << "===========================================\n\n";

    // Test various sizes (all multiples of tile size)
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {128, 128, 128},    // Small
        {256, 256, 256},    // Medium
        {512, 512, 512},    // Large
        {1024, 1024, 1024}, // Very large
        {128, 512, 256},    // Non-square
        {512, 128, 384},    // Non-square
    };

    std::vector<TestResult> results;
    int num_passed = 0;

    for(const auto& [M, N, K] : test_sizes)
    {
        std::cout << "Testing M=" << M << " N=" << N << " K=" << K << "...\n";

        auto result = run_test(dispatcher, M, N, K);
        results.push_back(result);

        std::cout << "  Time: " << result.time_ms << " ms\n";
        std::cout << "  Performance: " << result.tflops << " TFLOPS\n";
        std::cout << "  Accuracy: " << (100.0f * result.correct / result.total) << "%\n";
        std::cout << "  Status: " << (result.passed ? "[OK] PASS" : "[FAIL] FAIL") << "\n\n";

        if(result.passed)
            num_passed++;
    }

    // Summary
    std::cout << "===========================================\n";
    std::cout << "Summary\n";
    std::cout << "===========================================\n\n";

    std::cout << "Results by size:\n";
    std::cout << "  Size           | Time (ms) | TFLOPS | Accuracy | Status\n";
    std::cout << "  ---------------|-----------|--------|----------|--------\n";

    for(const auto& r : results)
    {
        char size_str[32];
        snprintf(size_str, sizeof(size_str), "%4dx%4dx%4d", r.M, r.N, r.K);

        printf("  %-14s | %9.4f | %6.2f | %7.2f%% | %s\n",
               size_str,
               r.time_ms,
               r.tflops,
               100.0f * r.correct / r.total,
               r.passed ? "[OK]" : "[FAIL]");
    }

    std::cout << "\n";
    std::cout << "Tests passed: " << num_passed << "/" << results.size() << "\n";

    if(num_passed == results.size())
    {
        std::cout << "\n[OK] ALL TESTS PASSED\n";
        return 0;
    }
    else
    {
        std::cout << "\n[FAIL] SOME TESTS FAILED\n";
        return 1;
    }
}
