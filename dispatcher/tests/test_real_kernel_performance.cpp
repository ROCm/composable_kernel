// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Performance test with real GPU kernel
 * Measures and reports detailed performance metrics
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

int main()
{
    std::cout << "=======================================\n";
    std::cout << "Performance Test - Real GPU Kernel\n";
    std::cout << "=======================================\n\n";

    std::cout << "Kernel: " << KERNEL_NAME << "\n";
    std::cout << "Device: AMD Instinct MI325X (gfx942)\n\n";

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

    // Performance benchmark sizes
    std::vector<std::tuple<int, int, int, const char*>> benchmarks = {
        {128, 128, 128, "Tiny"},
        {256, 256, 256, "Small"},
        {512, 512, 512, "Medium"},
        {1024, 1024, 1024, "Large"},
        {2048, 2048, 2048, "Very Large"},
    };

    std::cout << "Performance Benchmark Results\n";
    std::cout << "=============================\n\n";

    std::cout << "  Size      | Time (ms) | TFLOPS | BW (GB/s) | Status\n";
    std::cout << "  ----------|-----------|--------|-----------|--------\n";

    bool all_passed = true;

    for(const auto& [M, N, K, label] : benchmarks)
    {
        // Prepare data
        std::vector<ADataType> A_host(M * K, ADataType(1.0f));
        std::vector<BDataType> B_host(K * N, BDataType(1.0f));
        std::vector<CDataType> C_gpu(M * N);

        ADataType *A_dev, *B_dev;
        CDataType* C_dev;

        HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(ADataType)));
        HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(BDataType)));
        HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(CDataType)));

        HIP_CHECK(
            hipMemcpy(A_dev, A_host.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(B_dev, B_host.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(CDataType)));

        // Execute
        Problem problem(M, N, K);
        float time_ms = dispatcher.run(A_dev, B_dev, C_dev, problem);

        // Calculate metrics
        double flops  = 2.0 * M * N * K;
        double tflops = (flops / (time_ms * 1e-3)) / 1e12;

        // Bandwidth (A + B read, C write)
        double bytes         = (M * K + K * N + M * N) * sizeof(CDataType);
        double bandwidth_gbs = (bytes / (time_ms * 1e-3)) / 1e9;

        // Validate
        HIP_CHECK(hipMemcpy(C_gpu.data(), C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));

        int correct = 0;
        for(int i = 0; i < M * N; i++)
        {
            if(std::abs(float(C_gpu[i]) - float(K)) < 1.0f)
                correct++;
        }

        bool passed = (correct == M * N);
        all_passed  = all_passed && passed;

        char size_label[32];
        snprintf(size_label, sizeof(size_label), "%s %d^3", label, M);

        printf("  %-9s | %9.4f | %6.2f | %9.1f | %s\n",
               size_label,
               time_ms,
               tflops,
               bandwidth_gbs,
               passed ? "[OK]" : "[FAIL]");

        HIP_CHECK(hipFree(A_dev));
        HIP_CHECK(hipFree(B_dev));
        HIP_CHECK(hipFree(C_dev));
    }

    std::cout << "\n";

    if(all_passed)
    {
        std::cout << "[OK] ALL PERFORMANCE TESTS PASSED\n";
        return 0;
    }
    else
    {
        std::cout << "[FAIL] SOME TESTS FAILED\n";
        return 1;
    }
}
