// SPDX-License-Identifier: MIT
// Verify data flows correctly between CPU and GPU

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/host/check_err.hpp"
#include <hip/hip_runtime.h>
#include <iostream>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;

#define HIP_CHECK(call) { hipError_t err = call; if(err != hipSuccess) exit(1); }

// Calculate error thresholds - from tile_engine gemm_benchmark.hpp
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

int main()
{
    std::cout << "======================================================================\n";
    std::cout << "Data Flow Verification Test\n";
    std::cout << "======================================================================\n\n";
    
    const int M = 256, N = 256, K = 256;
    
    // Step 1: Create and initialize host tensors
    std::cout << "Step 1: Creating host tensors with layout descriptors...\n";
    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, K, ck_tile::bool_constant<true>{}));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, K, ck_tile::bool_constant<false>{}));
    ck_tile::HostTensor<CDataType> c_cpu_ref({M, N});
    ck_tile::HostTensor<CDataType> c_gpu_result({M, N});
    
    std::srand(12345);
    for(std::size_t i = 0; i < a_m_k.get_element_space_size(); i++) {
        a_m_k.mData[i] = ADataType(float(rand()) / RAND_MAX);
    }
    for(std::size_t i = 0; i < b_k_n.get_element_space_size(); i++) {
        b_k_n.mData[i] = BDataType(float(rand()) / RAND_MAX);
    }
    c_cpu_ref.SetZero();
    c_gpu_result.SetZero();
    
    std::cout << "  OK Initialized " << M*K + K*N << " values\n";
    std::cout << "  A sample values: " << float(a_m_k.mData[0]) << ", " 
              << float(a_m_k.mData[1]) << ", " << float(a_m_k.mData[2]) << "\n";
    std::cout << "  B sample values: " << float(b_k_n.mData[0]) << ", " 
              << float(b_k_n.mData[1]) << ", " << float(b_k_n.mData[2]) << "\n\n";
    
    // Step 2: Compute CPU reference
    std::cout << "Step 2: Computing CPU reference...\n";
    ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
        a_m_k, b_k_n, c_cpu_ref);
    
    std::cout << "  OK CPU result computed\n";
    std::cout << "  CPU C sample: " << float(c_cpu_ref.mData[0]) << ", " 
              << float(c_cpu_ref.mData[1]) << ", " << float(c_cpu_ref.mData[2]) << "\n\n";
    
    // Step 3: Copy SAME data to GPU
    std::cout << "Step 3: Copying SAME data to GPU...\n";
    ADataType *a_dev, *b_dev;
    CDataType *c_dev;
    HIP_CHECK(hipMalloc(&a_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&b_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&c_dev, M * N * sizeof(CDataType)));
    
    std::cout << "  Copying from a_m_k.data() = " << (void*)a_m_k.data() 
              << " (size=" << M*K*sizeof(ADataType) << ")\n";
    std::cout << "  Copying from b_k_n.data() = " << (void*)b_k_n.data() 
              << " (size=" << K*N*sizeof(BDataType) << ")\n";
    
    HIP_CHECK(hipMemcpy(a_dev, a_m_k.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_dev, b_k_n.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(c_dev, 0, M * N * sizeof(CDataType)));
    
    // Verify data copied correctly by copying back
    std::vector<ADataType> a_verify(M * K);
    std::vector<BDataType> b_verify(K * N);
    HIP_CHECK(hipMemcpy(a_verify.data(), a_dev, M * K * sizeof(ADataType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(b_verify.data(), b_dev, K * N * sizeof(BDataType), hipMemcpyDeviceToHost));
    
    int a_match = 0, b_match = 0;
    for(size_t i = 0; i < a_m_k.get_element_space_size(); i++) {
        if(a_m_k.mData[i] == a_verify[i]) a_match++;
    }
    for(size_t i = 0; i < b_k_n.get_element_space_size(); i++) {
        if(b_k_n.mData[i] == b_verify[i]) b_match++;
    }
    
    std::cout << "  OK Data copied to GPU\n";
    std::cout << "  Verification: A " << a_match << "/" << M*K << " match (" 
              << (100.0f*a_match/(M*K)) << "%)\n";
    std::cout << "  Verification: B " << b_match << "/" << K*N << " match (" 
              << (100.0f*b_match/(K*N)) << "%)\n\n";
    
    if(a_match != M*K || b_match != K*N) {
        std::cout << "  [FAIL] DATA TRANSFER ISSUE!\n";
        return 1;
    }
    
    // Step 4: Execute on GPU
    std::cout << "Step 4: Executing on GPU via dispatcher...\n";
    
    // Create kernel
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
    Problem problem(M, N, K);
    
    float gpu_time = dispatcher.run(a_dev, b_dev, c_dev, problem, nullptr);
    
    std::cout << "  OK GPU executed: " << gpu_time << " ms\n";
    
    // Copy GPU result back
    HIP_CHECK(hipMemcpy(c_gpu_result.data(), c_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));
    std::cout << "  GPU C sample: " << float(c_gpu_result.mData[0]) << ", " 
              << float(c_gpu_result.mData[1]) << ", " << float(c_gpu_result.mData[2]) << "\n\n";
    
    // Step 5: Compare
    std::cout << "Step 5: Comparing results...\n";
    std::cout << "  CPU reference: " << float(c_cpu_ref.mData[0]) << ", " 
              << float(c_cpu_ref.mData[1]) << ", " << float(c_cpu_ref.mData[2]) << "\n";
    std::cout << "  GPU result:    " << float(c_gpu_result.mData[0]) << ", " 
              << float(c_gpu_result.mData[1]) << ", " << float(c_gpu_result.mData[2]) << "\n\n";
    
    // Detailed comparison
    auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
        K, 1, *std::max_element(c_cpu_ref.mData.begin(), c_cpu_ref.mData.end()));
    
    bool pass = ck_tile::check_err(
        c_gpu_result, c_cpu_ref, "GPU vs CPU", 
        rtol_atol.at(ck_tile::number<0>{}), rtol_atol.at(ck_tile::number<1>{}));
    
    HIP_CHECK(hipFree(a_dev));
    HIP_CHECK(hipFree(b_dev));
    HIP_CHECK(hipFree(c_dev));
    
    std::cout << "======================================================================\n";
    if(pass) {
        std::cout << "[OK] DATA FLOW VERIFIED - Same input → Same output\n";
        std::cout << "[OK] CPU and GPU produce identical results\n";
    } else {
        std::cout << "[FAIL] Results differ (but data transfer is correct)\n";
    }
    std::cout << "======================================================================\n";
    
    return pass ? 0 : 1;
}

