// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GPU Helper - C++ executable for GPU GEMM execution
 *
 * A CLI tool for Python to execute GPU GEMM with generated kernels.
 * Usage: gpu_helper <M> <N> <K> [--validate]
 *
 * Kernel header included via -include flag at compile time.
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
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
            std::cerr << "HIP_ERROR: " << hipGetErrorString(err) << "\n"; \
            exit(1);                                                      \
        }                                                                 \
    }

// CPU reference GEMM (for validation)
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
                // A: RowMajor, B: ColumnMajor
                acc += float(A[m * K + k]) * float(B[k + n * K]);
            }
            C[m * N + n] = T(acc);
        }
    }
}

int main(int argc, char** argv)
{
    // Parse arguments
    if(argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> [--validate]\n";
        std::cerr << "\nOptions:\n";
        std::cerr << "  M, N, K    : Problem dimensions\n";
        std::cerr << "  --validate : Compare GPU results with CPU reference\n";
        return 1;
    }

    int M         = std::atoi(argv[1]);
    int N         = std::atoi(argv[2]);
    int K         = std::atoi(argv[3]);
    bool validate = (argc > 4 && std::string(argv[4]) == "--validate");

    // Output in JSON-like format for easy Python parsing
    std::cout << "{" << std::endl;
    std::cout << "  \"problem\": {\"M\": " << M << ", \"N\": " << N << ", \"K\": " << K << "},"
              << std::endl;
    std::cout << "  \"kernel\": \"" << KERNEL_NAME << "\"," << std::endl;

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
    Problem problem(M, N, K);

    auto selected = dispatcher.select_kernel(problem);
    if(!selected)
    {
        std::cout << "  \"error\": \"No kernel selected\"" << std::endl;
        std::cout << "}" << std::endl;
        return 1;
    }

    std::cout << "  \"selected_kernel\": \"" << selected->get_name() << "\"," << std::endl;

    // Prepare data: A=1, B=1, so C should be K
    std::vector<ADataType> A_host(M * K, ADataType(1.0f));
    std::vector<BDataType> B_host(K * N, BDataType(1.0f));
    std::vector<CDataType> C_gpu(M * N);

    // GPU execution
    ADataType *A_dev, *B_dev;
    CDataType* C_dev;

    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(CDataType)));

    HIP_CHECK(hipMemcpy(A_dev, A_host.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B_host.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(CDataType)));

    float gpu_time = dispatcher.run(A_dev, B_dev, C_dev, problem);

    HIP_CHECK(hipMemcpy(C_gpu.data(), C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));

    // Calculate performance
    double flops  = 2.0 * M * N * K;
    double tflops = (flops / (gpu_time * 1e-3)) / 1e12;

    std::cout << "  \"execution\": {" << std::endl;
    std::cout << "    \"time_ms\": " << gpu_time << "," << std::endl;
    std::cout << "    \"tflops\": " << tflops << "," << std::endl;
    std::cout << "    \"flops\": " << (long long)flops << std::endl;
    std::cout << "  }," << std::endl;

    // Validation
    if(validate)
    {
        std::vector<CDataType> C_cpu(M * N);
        cpu_gemm(A_host, B_host, C_cpu, M, N, K);

        int correct     = 0;
        float max_error = 0.0f;

        for(int i = 0; i < M * N; i++)
        {
            float gpu_val = float(C_gpu[i]);
            float cpu_val = float(C_cpu[i]);
            float error   = std::abs(gpu_val - cpu_val) / (std::abs(cpu_val) + 1e-5f);

            max_error = std::max(max_error, error);

            if(error < 0.02f)
            {
                correct++;
            }
        }

        float accuracy = 100.0f * correct / (M * N);

        std::cout << "  \"validation\": {" << std::endl;
        std::cout << "    \"accuracy\": " << accuracy << "," << std::endl;
        std::cout << "    \"max_error\": " << max_error << "," << std::endl;
        std::cout << "    \"correct_elements\": " << correct << "," << std::endl;
        std::cout << "    \"total_elements\": " << M * N << std::endl;
        std::cout << "  }," << std::endl;
    }

    std::cout << "  \"status\": \"success\"" << std::endl;
    std::cout << "}" << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));

    return 0;
}
