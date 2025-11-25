// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * CK Tile Dispatcher - Correctness Verification
 *
 * Uses CK Tile's reference_gemm to validate GPU results.
 * Follows tile_engine validation pattern.
 */

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/host/check_err.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;

#define HIP_CHECK(call)                                                   \
    {                                                                     \
        hipError_t err = call;                                            \
        if(err != hipSuccess)                                             \
        {                                                                 \
            std::cerr << "HIP Error: " << hipGetErrorString(err) << "\n"; \
            exit(1);                                                      \
        }                                                                 \
    }

// Calculate error thresholds - EXACT copy from tile_engine gemm_benchmark.hpp
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

    // Calculate thresholds using CK Tile's type-aware functions
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);

    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

int main(int argc, char** argv)
{
    std::cout << "======================================================================\n";
    std::cout << "CK Tile Dispatcher - Correctness Verification\n";
    std::cout << "Uses CK Tile reference_gemm for validation\n";
    std::cout << "======================================================================\n\n";

    // Parse problem size
    int M = 256, N = 256, K = 256;
    if(argc >= 4)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    std::cout << "Problem: M=" << M << " N=" << N << " K=" << K << "\n\n";

    // Create kernel key
    KernelKey key;
    key.signature.dtype_a        = DataType::FP16;
    key.signature.dtype_b        = DataType::FP16;
    key.signature.dtype_c        = DataType::FP16;
    key.signature.dtype_acc      = DataType::FP32;
    key.signature.layout_a       = LayoutTag::RowMajor;
    key.signature.layout_b       = LayoutTag::ColMajor;
    key.signature.layout_c       = LayoutTag::RowMajor;
    key.signature.elementwise_op = "PassThrough";
    key.signature.num_d_tensors  = 0;
    key.signature.split_k        = 1;

    key.algorithm.tile_shape      = {128, 128, 64};
    key.algorithm.wave_shape      = {2, 2, 1};
    key.algorithm.warp_tile_shape = {32, 32, 16};
    key.algorithm.pipeline        = Pipeline::CompV4;
    key.algorithm.scheduler       = Scheduler::Intrawave;
    key.algorithm.epilogue        = Epilogue::CShuffle;
    key.algorithm.block_size      = 256;
    key.algorithm.double_buffer   = true;
    key.algorithm.persistent      = false;
    key.gfx_arch                  = "gfx942";

    // Register kernel
    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, std::string(KERNEL_NAME));

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel);

    Dispatcher dispatcher;
    Problem problem(M, N, K);

    // Step 1: Create host tensors with correct layouts (matching tile_engine)
    std::cout << "Step 1: Creating tensors with correct layout descriptors...\n";

    // Use host_tensor_descriptor with strides (like tile_engine does)
    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, K, ck_tile::bool_constant<true>{})); // Row-major
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, K, ck_tile::bool_constant<false>{})); // Column-major
    ck_tile::HostTensor<CDataType> c_m_n_gpu_result(
        ck_tile::host_tensor_descriptor(M, N, N, ck_tile::bool_constant<true>{})); // Row-major
    ck_tile::HostTensor<CDataType> c_m_n_cpu_reference(
        ck_tile::host_tensor_descriptor(M, N, N, ck_tile::bool_constant<true>{})); // Row-major

    // Initialize with random data
    std::srand(54321); // Fixed seed

    for(std::size_t i = 0; i < a_m_k.get_element_space_size(); i++)
    {
        a_m_k.mData[i] = ADataType((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f);
    }

    for(std::size_t i = 0; i < b_k_n.get_element_space_size(); i++)
    {
        b_k_n.mData[i] = BDataType((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f);
    }

    c_m_n_gpu_result.SetZero();
    c_m_n_cpu_reference.SetZero();

    std::cout << "  OK Initialized random data\n\n";

    // Step 2: Compute CPU reference using CK Tile reference_gemm
    std::cout << "Step 2: Computing CPU reference (ck_tile::reference_gemm)...\n";

    ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
        a_m_k, b_k_n, c_m_n_cpu_reference);

    std::cout << "  OK CPU reference computed\n";
    std::cout << "  Reference range: [" << float(c_m_n_cpu_reference.mData.front()) << ", "
              << float(c_m_n_cpu_reference.mData.back()) << "]\n\n";

    // Step 3: Execute on GPU via dispatcher
    std::cout << "Step 3: Executing on GPU via dispatcher...\n";

    // Allocate device memory
    ADataType *a_dev, *b_dev;
    CDataType* c_dev;
    HIP_CHECK(hipMalloc(&a_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&b_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&c_dev, M * N * sizeof(CDataType)));

    // Copy to device
    HIP_CHECK(hipMemcpy(a_dev, a_m_k.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_dev, b_k_n.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(c_dev, 0, M * N * sizeof(CDataType)));

    // Execute
    float gpu_time = dispatcher.run(a_dev, b_dev, c_dev, problem, nullptr);

    // Copy result back
    HIP_CHECK(hipMemcpy(
        c_m_n_gpu_result.data(), c_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));

    float tflops = (2.0f * M * N * K) / (gpu_time * 1e9);
    std::cout << "  OK GPU execution: " << gpu_time << " ms / " << tflops << " TFLOPS\n\n";

    // Step 4: Validate using CK Tile check_err
    std::cout << "Step 4: Validating results (ck_tile::check_err)...\n";

    // Calculate error thresholds using tile_engine logic
    const float max_accumulated_value =
        *std::max_element(c_m_n_cpu_reference.mData.begin(), c_m_n_cpu_reference.mData.end());

    auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
        K, 1, max_accumulated_value);

    float rtol = rtol_atol.at(ck_tile::number<0>{});
    float atol = rtol_atol.at(ck_tile::number<1>{});

    std::cout << "  Relative error threshold: " << rtol << "\n";
    std::cout << "  Absolute error threshold: " << atol << "\n";

    bool pass =
        ck_tile::check_err(c_m_n_gpu_result, c_m_n_cpu_reference, "GPU vs CPU results", rtol, atol);

    std::cout << "  Verification result: " << (pass ? "CORRECT" : "FAILED") << "\n\n";

    // Cleanup
    HIP_CHECK(hipFree(a_dev));
    HIP_CHECK(hipFree(b_dev));
    HIP_CHECK(hipFree(c_dev));

    // Final summary
    std::cout << "======================================================================\n";
    if(pass)
    {
        std::cout << "[OK] VALIDATION PASSED - GPU results are correct!\n";
        std::cout << "======================================================================\n";
        std::cout << "\nSummary:\n";
        std::cout << "  Problem: " << M << "x" << N << "x" << K << "\n";
        std::cout << "  GPU Performance: " << gpu_time << " ms / " << tflops << " TFLOPS\n";
        std::cout << "  Correctness: [OK] VERIFIED (matches CPU reference)\n";
        std::cout << "  Tolerance: rtol=" << rtol << ", atol=" << atol << "\n";
        std::cout << "\n[OK] Dispatcher executes correct GEMM!\n";
        return 0;
    }
    else
    {
        std::cout << "[FAIL] VALIDATION FAILED - Results do not match!\n";
        std::cout << "======================================================================\n";
        return 1;
    }
}
