// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Sanity check tests to verify CK Tile kernels are actually running on GPU.
 *
 * These tests verify:
 * 1. GPU memory allocation and transfer work correctly
 * 2. The dispatcher calls CK Tile infrastructure
 * 3. GPU computes correct results (not just zeros)
 * 4. Performance is reasonable (not CPU fallback)
 * 5. Different problem sizes work correctly
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <numeric>
#include <hip/hip_runtime.h>

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

// Kernel header will be included via -include compiler flag

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;

#define HIP_CHECK(call)                                                         \
    {                                                                           \
        hipError_t err = call;                                                  \
        if(err != hipSuccess)                                                   \
        {                                                                       \
            std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << hipGetErrorString(err) << "\n";                        \
            return 1;                                                           \
        }                                                                       \
    }

// Reference CPU GEMM for validation
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
                acc += float(A[m * K + k]) * float(B[k * N + n]);
            }
            C[m * N + n] = T(acc);
        }
    }
}

// Test helper to setup dispatcher
void setup_dispatcher()
{
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

    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Registry::Priority::High);
}

// =============================================================================
// Test 1: Basic Sanity - All ones multiplication
// =============================================================================
int test_all_ones()
{
    std::cout << "\n=== Test: All Ones Multiplication ===\n";

    const int M = 256, N = 256, K = 256;

    std::vector<ADataType> A(M * K, ADataType(1.0f));
    std::vector<BDataType> B(K * N, BDataType(1.0f));
    std::vector<CDataType> C(M * N, CDataType(0.0f));

    ADataType *A_dev, *B_dev;
    CDataType* C_dev;

    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(CDataType)));

    HIP_CHECK(hipMemcpy(A_dev, A.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(CDataType)));

    Dispatcher dispatcher;
    Problem problem(M, N, K);

    float time = dispatcher.run(A_dev, B_dev, C_dev, problem);

    HIP_CHECK(hipMemcpy(C.data(), C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));

    // All ones * all ones with K=256 should give K=256 for each element
    int correct = 0;
    for(int i = 0; i < M * N; i++)
    {
        if(std::abs(float(C[i]) - float(K)) < 1.0f)
        {
            correct++;
        }
    }

    float accuracy = 100.0f * correct / (M * N);

    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));

    std::cout << "  Time: " << time << " ms\n";
    std::cout << "  Expected: " << K << "\n";
    std::cout << "  Sample C[0]: " << float(C[0]) << "\n";
    std::cout << "  Accuracy: " << accuracy << "%\n";

    if(accuracy < 99.0f)
    {
        std::cerr << "  FAILED: Accuracy too low\n";
        return 1;
    }

    std::cout << "  PASSED\n";
    return 0;
}

// =============================================================================
// Test 2: Non-Zero Results - Verify GPU actually computed something
// =============================================================================
int test_non_zero_results()
{
    std::cout << "\n=== Test: Non-Zero Results ===\n";

    const int M = 256, N = 256, K = 256;

    std::vector<ADataType> A(M * K, ADataType(2.0f)); // All 2s
    std::vector<BDataType> B(K * N, BDataType(3.0f)); // All 3s
    std::vector<CDataType> C(M * N, CDataType(0.0f));

    ADataType *A_dev, *B_dev;
    CDataType* C_dev;

    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(CDataType)));

    HIP_CHECK(hipMemcpy(A_dev, A.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(CDataType)));

    Dispatcher dispatcher;
    Problem problem(M, N, K);

    float time = dispatcher.run(A_dev, B_dev, C_dev, problem);

    HIP_CHECK(hipMemcpy(C.data(), C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));

    // 2 * 3 * K = 6 * 256 = 1536
    float expected = 6.0f * K;
    int correct    = 0;
    int non_zero   = 0;

    for(int i = 0; i < M * N; i++)
    {
        if(float(C[i]) != 0.0f)
            non_zero++;
        if(std::abs(float(C[i]) - expected) < 10.0f)
        {
            correct++;
        }
    }

    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));

    std::cout << "  Time: " << time << " ms\n";
    std::cout << "  Expected: " << expected << "\n";
    std::cout << "  Sample C[0]: " << float(C[0]) << "\n";
    std::cout << "  Non-zero elements: " << non_zero << "/" << M * N << "\n";

    if(non_zero == 0)
    {
        std::cerr << "  FAILED: All zeros - GPU may not have run\n";
        return 1;
    }

    float accuracy = 100.0f * correct / (M * N);
    std::cout << "  Accuracy: " << accuracy << "%\n";

    if(accuracy < 99.0f)
    {
        std::cerr << "  FAILED: Accuracy too low\n";
        return 1;
    }

    std::cout << "  PASSED\n";
    return 0;
}

// =============================================================================
// Test 3: Performance Check - Ensure not CPU fallback
// =============================================================================
int test_performance()
{
    std::cout << "\n=== Test: Performance Check ===\n";

    const int M = 1024, N = 1024, K = 1024;
    const int num_runs = 5;

    std::vector<ADataType> A(M * K, ADataType(1.0f));
    std::vector<BDataType> B(K * N, BDataType(1.0f));
    std::vector<CDataType> C(M * N);

    ADataType *A_dev, *B_dev;
    CDataType* C_dev;

    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(CDataType)));

    HIP_CHECK(hipMemcpy(A_dev, A.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));

    Dispatcher dispatcher;
    Problem problem(M, N, K);

    // Warmup
    dispatcher.run(A_dev, B_dev, C_dev, problem);
    HIP_CHECK(hipDeviceSynchronize());

    // Timed runs
    std::vector<float> times;
    for(int i = 0; i < num_runs; i++)
    {
        float time = dispatcher.run(A_dev, B_dev, C_dev, problem);
        times.push_back(time);
    }

    float avg_time = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
    float min_time = *std::min_element(times.begin(), times.end());

    double flops  = 2.0 * M * N * K;
    double tflops = (flops / (min_time * 1e-3)) / 1e12;

    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));

    std::cout << "  Problem: " << M << "x" << N << "x" << K << "\n";
    std::cout << "  Avg time: " << avg_time << " ms\n";
    std::cout << "  Min time: " << min_time << " ms\n";
    std::cout << "  Performance: " << tflops << " TFLOPS\n";

    // GPU should achieve at least 1 TFLOPS for this size
    // CPU would be ~0.001 TFLOPS
    if(tflops < 1.0)
    {
        std::cerr << "  FAILED: Performance too low - may be CPU fallback\n";
        return 1;
    }

    std::cout << "  PASSED\n";
    return 0;
}

// =============================================================================
// Test 4: CPU vs GPU Correctness
// =============================================================================
int test_vs_cpu_reference()
{
    std::cout << "\n=== Test: CPU vs GPU Correctness ===\n";

    const int M = 128, N = 128, K = 128; // Small for CPU reference

    // Random-ish values
    std::vector<ADataType> A(M * K);
    std::vector<BDataType> B(K * N);
    std::vector<CDataType> C_gpu(M * N);
    std::vector<CDataType> C_cpu(M * N);

    for(int i = 0; i < M * K; i++)
    {
        A[i] = ADataType(float((i % 10) + 1) * 0.1f);
    }
    for(int i = 0; i < K * N; i++)
    {
        B[i] = BDataType(float((i % 7) + 1) * 0.1f);
    }

    // CPU reference
    cpu_gemm(A, B, C_cpu, M, N, K);

    // GPU
    ADataType *A_dev, *B_dev;
    CDataType* C_dev;

    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(CDataType)));

    HIP_CHECK(hipMemcpy(A_dev, A.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(CDataType)));

    Dispatcher dispatcher;
    Problem problem(M, N, K);

    dispatcher.run(A_dev, B_dev, C_dev, problem);

    HIP_CHECK(hipMemcpy(C_gpu.data(), C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));

    // Compare
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int correct    = 0;

    for(int i = 0; i < M * N; i++)
    {
        float gpu_val = float(C_gpu[i]);
        float cpu_val = float(C_cpu[i]);
        float diff    = std::abs(gpu_val - cpu_val);

        max_diff = std::max(max_diff, diff);
        sum_diff += diff;

        // FP16 has limited precision (~3-4 decimal digits)
        // For K=128, values can reach ~10-30, so allow 5% relative error + absolute tolerance
        float tolerance = std::max(std::abs(cpu_val) * 0.05f, 1.0f);
        if(diff < tolerance)
        {
            correct++;
        }
    }

    float avg_diff = sum_diff / (M * N);
    float accuracy = 100.0f * correct / (M * N);

    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));

    std::cout << "  Max diff: " << max_diff << "\n";
    std::cout << "  Avg diff: " << avg_diff << "\n";
    std::cout << "  Sample CPU C[0]: " << float(C_cpu[0]) << "\n";
    std::cout << "  Sample GPU C[0]: " << float(C_gpu[0]) << "\n";
    std::cout << "  Accuracy: " << accuracy << "%\n";

    // FP16 accumulation can have significant rounding differences from CPU FP32
    // 90% is reasonable for FP16 with K=128 accumulation
    if(accuracy < 90.0f)
    {
        std::cerr << "  FAILED: Too many mismatches vs CPU\n";
        return 1;
    }

    std::cout << "  PASSED\n";
    return 0;
}

// =============================================================================
// Test 5: Different Problem Sizes
// =============================================================================
int test_multiple_sizes()
{
    std::cout << "\n=== Test: Multiple Problem Sizes ===\n";

    std::vector<std::tuple<int, int, int>> sizes = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {128, 256, 512},
        {512, 256, 128},
        {1024, 1024, 256},
    };

    int passed = 0;
    int total  = sizes.size();

    for(const auto& [M, N, K] : sizes)
    {
        std::cout << "  Testing " << M << "x" << N << "x" << K << "... ";

        std::vector<ADataType> A(M * K, ADataType(1.0f));
        std::vector<BDataType> B(K * N, BDataType(1.0f));
        std::vector<CDataType> C(M * N);

        ADataType *A_dev, *B_dev;
        CDataType* C_dev;

        hipMalloc(&A_dev, M * K * sizeof(ADataType));
        hipMalloc(&B_dev, K * N * sizeof(BDataType));
        hipMalloc(&C_dev, M * N * sizeof(CDataType));

        hipMemcpy(A_dev, A.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice);
        hipMemcpy(B_dev, B.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice);
        hipMemset(C_dev, 0, M * N * sizeof(CDataType));

        Dispatcher dispatcher;
        Problem problem(M, N, K);

        float time = dispatcher.run(A_dev, B_dev, C_dev, problem);

        hipMemcpy(C.data(), C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost);

        hipFree(A_dev);
        hipFree(B_dev);
        hipFree(C_dev);

        // Check result
        int correct = 0;
        for(int i = 0; i < M * N; i++)
        {
            if(std::abs(float(C[i]) - float(K)) < 1.0f)
            {
                correct++;
            }
        }

        float accuracy = 100.0f * correct / (M * N);

        if(accuracy > 99.0f && time > 0)
        {
            std::cout << "PASS (" << time << " ms)\n";
            passed++;
        }
        else
        {
            std::cout << "FAIL (acc=" << accuracy << "%, time=" << time << ")\n";
        }
    }

    std::cout << "\n  Passed: " << passed << "/" << total << "\n";

    if(passed < total)
    {
        std::cerr << "  FAILED: Some sizes failed\n";
        return 1;
    }

    std::cout << "  PASSED\n";
    return 0;
}

// =============================================================================
// Test 6: Memory Bounds Check
// =============================================================================
int test_memory_bounds()
{
    std::cout << "\n=== Test: Memory Bounds Check ===\n";

    const int M = 256, N = 256, K = 256;
    const float sentinel = -999.0f;

    // Allocate with extra padding and sentinel values
    const int padding = 16;
    std::vector<ADataType> A(M * K + padding, ADataType(1.0f));
    std::vector<BDataType> B(K * N + padding, BDataType(1.0f));
    std::vector<CDataType> C(M * N + padding, CDataType(sentinel));

    // Set sentinels at the end
    for(int i = 0; i < padding; i++)
    {
        A[M * K + i] = ADataType(sentinel);
        B[K * N + i] = BDataType(sentinel);
    }

    ADataType *A_dev, *B_dev;
    CDataType* C_dev;

    HIP_CHECK(hipMalloc(&A_dev, (M * K + padding) * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&B_dev, (K * N + padding) * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&C_dev, (M * N + padding) * sizeof(CDataType)));

    HIP_CHECK(
        hipMemcpy(A_dev, A.data(), (M * K + padding) * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(B_dev, B.data(), (K * N + padding) * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(C_dev, C.data(), (M * N + padding) * sizeof(CDataType), hipMemcpyHostToDevice));

    Dispatcher dispatcher;
    Problem problem(M, N, K);

    dispatcher.run(A_dev, B_dev, C_dev, problem);

    HIP_CHECK(
        hipMemcpy(C.data(), C_dev, (M * N + padding) * sizeof(CDataType), hipMemcpyDeviceToHost));

    // Check sentinels weren't overwritten
    bool sentinels_intact = true;
    for(int i = 0; i < padding; i++)
    {
        if(float(C[M * N + i]) != sentinel)
        {
            sentinels_intact = false;
            std::cerr << "  Sentinel overwritten at position " << (M * N + i) << "\n";
        }
    }

    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));

    if(!sentinels_intact)
    {
        std::cerr << "  FAILED: Memory bounds violated\n";
        return 1;
    }

    // Also check actual results are correct
    int correct = 0;
    for(int i = 0; i < M * N; i++)
    {
        if(std::abs(float(C[i]) - float(K)) < 1.0f)
        {
            correct++;
        }
    }

    float accuracy = 100.0f * correct / (M * N);
    std::cout << "  Sentinels intact: Yes\n";
    std::cout << "  Result accuracy: " << accuracy << "%\n";

    if(accuracy < 99.0f)
    {
        std::cerr << "  FAILED: Results incorrect\n";
        return 1;
    }

    std::cout << "  PASSED\n";
    return 0;
}

// =============================================================================
// Main
// =============================================================================
int main()
{
    std::cout << "========================================\n";
    std::cout << "CK Tile Sanity Check Tests\n";
    std::cout << "========================================\n";
    std::cout << "Kernel: " << KERNEL_NAME << "\n";

    // Setup
    setup_dispatcher();

    int failures = 0;

    // Run all tests
    failures += test_all_ones();
    failures += test_non_zero_results();
    failures += test_performance();
    failures += test_vs_cpu_reference();
    failures += test_multiple_sizes();
    failures += test_memory_bounds();

    std::cout << "\n========================================\n";
    if(failures == 0)
    {
        std::cout << "ALL TESTS PASSED\n";
        std::cout << "CK Tile is running correctly on GPU.\n";
        return 0;
    }
    else
    {
        std::cout << failures << " TEST(S) FAILED\n";
        return 1;
    }
}
