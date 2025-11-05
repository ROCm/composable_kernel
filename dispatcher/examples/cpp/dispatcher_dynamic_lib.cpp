// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Dispatcher Dynamic Library - For Python ctypes loading
 * 
 * This creates a .so that Python can load via ctypes.
 * Exposes simple C ABI for passing NumPy array pointers.
 * 
 * Kernel header included via -include at compile time.
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

// Kernel header included via -include compiler flag
// Defines: ADataType, BDataType, CDataType, AccDataType, SelectedKernel, KERNEL_NAME

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using Priority = ck_tile::dispatcher::Registry::Priority;

// Global dispatcher (initialized once)
static Dispatcher* g_dispatcher = nullptr;
static bool g_initialized = false;

#define HIP_CHECK(call) { \
    hipError_t err = call; \
    if(err != hipSuccess) { \
        return -1; \
    } \
}

extern "C" {

/**
 * Initialize dispatcher with a kernel
 * Must be called before run_gemm
 * 
 * Returns: 0 on success, -1 on error
 */
int dispatcher_initialize() {
    if (g_initialized) {
        return 0;  // Already initialized
    }
    
    // Create kernel key
    KernelKey key;
    key.signature.dtype_a = DataType::FP16;
    key.signature.dtype_b = DataType::FP16;
    key.signature.dtype_c = DataType::FP16;
    key.signature.dtype_acc = DataType::FP32;
    key.signature.layout_a = LayoutTag::RowMajor;
    key.signature.layout_b = LayoutTag::ColMajor;
    key.signature.layout_c = LayoutTag::RowMajor;
    key.signature.transpose_a = false;
    key.signature.transpose_b = false;
    key.signature.grouped = false;
    key.signature.split_k = 1;
    key.signature.elementwise_op = "PassThrough";
    key.signature.num_d_tensors = 0;
    key.signature.structured_sparsity = false;
    
    key.algorithm.tile_shape = {128, 128, 32};
    key.algorithm.wave_shape = {2, 2, 1};
    key.algorithm.warp_tile_shape = {32, 32, 16};
    key.algorithm.pipeline = Pipeline::CompV4;
    key.algorithm.scheduler = Scheduler::Intrawave;
    key.algorithm.epilogue = Epilogue::CShuffle;
    key.algorithm.block_size = 256;
    key.algorithm.double_buffer = true;
    key.algorithm.persistent = false;
    key.algorithm.preshuffle = false;
    key.algorithm.transpose_c = false;
    key.algorithm.num_wave_groups = 1;
    key.gfx_arch = 942;
    
    // Register kernel
    auto kernel = create_generated_tile_kernel<
        SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(key, KERNEL_NAME);
    
    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Priority::High);
    
    // Create dispatcher
    g_dispatcher = new Dispatcher();
    g_initialized = true;
    
    return 0;
}

/**
 * Get the selected kernel name for a problem
 * 
 * Args:
 *   M, N, K: Problem dimensions
 *   name_buffer: Output buffer for kernel name (at least 256 bytes)
 *   buffer_size: Size of name_buffer
 * 
 * Returns: 0 on success, -1 on error
 */
int dispatcher_select_kernel(
    int64_t M, int64_t N, int64_t K,
    char* name_buffer, int buffer_size)
{
    if (!g_initialized) {
        return -1;
    }
    
    Problem problem(M, N, K);
    auto kernel = g_dispatcher->select_kernel(problem);
    
    if (!kernel) {
        return -1;
    }
    
    std::string name = kernel->get_name();
    strncpy(name_buffer, name.c_str(), buffer_size - 1);
    name_buffer[buffer_size - 1] = '\0';
    
    return 0;
}

/**
 * Run GEMM on GPU via dispatcher
 * 
 * Args:
 *   A: Pointer to A matrix (M x K, row-major, float16)
 *   B: Pointer to B matrix (K x N, column-major, float16)
 *   C: Pointer to C matrix (M x N, row-major, float16) - OUTPUT
 *   M, N, K: Problem dimensions
 *   time_ms: Output pointer for execution time
 * 
 * Returns: 0 on success, -1 on error
 * 
 * Note: This function:
 * 1. Allocates GPU memory
 * 2. Copies A, B to GPU
 * 3. Runs dispatcher GEMM
 * 4. Copies C back to CPU
 * 5. Frees GPU memory
 */
int dispatcher_run_gemm(
    const void* A,    // Host pointer
    const void* B,    // Host pointer
    void* C,          // Host pointer (output)
    int64_t M,
    int64_t N,
    int64_t K,
    float* time_ms)   // Output
{
    if (!g_initialized || !A || !B || !C) {
        return -1;
    }
    
    // Cast to correct types
    const ADataType* A_host = static_cast<const ADataType*>(A);
    const BDataType* B_host = static_cast<const BDataType*>(B);
    CDataType* C_host = static_cast<CDataType*>(C);
    
    // Allocate GPU memory
    ADataType* A_dev = nullptr;
    BDataType* B_dev = nullptr;
    CDataType* C_dev = nullptr;
    
    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(CDataType)));
    
    // Copy input data to GPU
    HIP_CHECK(hipMemcpy(A_dev, A_host, M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B_host, K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(CDataType)));
    
    // Run GEMM via dispatcher
    Problem problem(M, N, K);
    float exec_time = g_dispatcher->run(A_dev, B_dev, C_dev, problem);
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(C_host, C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));
    
    // Store timing if requested
    if (time_ms) {
        *time_ms = exec_time;
    }
    
    // Cleanup GPU memory
    hipFree(A_dev);
    hipFree(B_dev);
    hipFree(C_dev);
    
    return 0;
}

/**
 * Get kernel information
 * 
 * Returns: Pointer to null-terminated kernel name string
 */
const char* dispatcher_get_kernel_name() {
    return KERNEL_NAME;
}

/**
 * Cleanup dispatcher resources
 */
void dispatcher_cleanup() {
    if (g_dispatcher) {
        delete g_dispatcher;
        g_dispatcher = nullptr;
    }
    g_initialized = false;
}

} // extern "C"

