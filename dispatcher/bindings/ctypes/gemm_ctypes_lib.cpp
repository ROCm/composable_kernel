// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * GEMM Dispatcher ctypes Library
 *
 * Provides C API for Python ctypes integration.
 * Kernel header included via -include at compile time.
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libdispatcher_gemm.so")
 *   lib.dispatcher_init()
 *   lib.dispatcher_run_gemm(...)
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

// Kernel header included via -include compiler flag
// Defines: ADataType, BDataType, CDataType, AccDataType, SelectedKernel, KERNEL_NAME

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using Priority = ck_tile::dispatcher::Registry::Priority;

// Global dispatcher (initialized once, managed via shared_ptr for safe cleanup)
static std::shared_ptr<Dispatcher> g_dispatcher = nullptr;
static bool g_initialized                       = false;

#define HIP_CHECK(call)        \
    {                          \
        hipError_t err = call; \
        if(err != hipSuccess)  \
        {                      \
            return -1;         \
        }                      \
    }

extern "C" {

/**
 * Initialize dispatcher with a kernel
 * Must be called before run_gemm
 *
 * Returns: 0 on success, -1 on error
 */
int dispatcher_initialize()
{
    if(g_initialized)
    {
        return 0; // Already initialized
    }

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

    // Register kernel
    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Priority::High);

    // Create dispatcher (using shared_ptr for safe memory management)
    g_dispatcher  = std::make_shared<Dispatcher>();
    g_initialized = true;

    return 0;
}

/**
 * Get kernel tile configuration
 *
 * Returns the block tile, warp tile, and wave configuration for the
 * registered kernel. This allows callers to understand dimension
 * requirements before attempting to run.
 *
 * Args:
 *   tile_m, tile_n, tile_k: Output for block tile dimensions
 *   warp_tile_m, warp_tile_n, warp_tile_k: Output for warp tile dimensions
 *   warp_m, warp_n, warp_k: Output for wave/warp configuration
 *
 * Returns: 0 on success, -1 if not initialized
 *
 * Note: For problem dimensions to be supported (without padding):
 *   - M must be divisible by tile_m
 *   - N must be divisible by tile_n
 *   - K must be divisible by tile_k
 */
int dispatcher_get_kernel_config(int* tile_m,
                                 int* tile_n,
                                 int* tile_k,
                                 int* warp_tile_m,
                                 int* warp_tile_n,
                                 int* warp_tile_k,
                                 int* warp_m,
                                 int* warp_n,
                                 int* warp_k)
{
    if(!g_initialized)
    {
        return -1;
    }

    auto kernels = Registry::instance().get_all();
    if(kernels.empty())
    {
        return -1;
    }

    // Get configuration from first kernel
    auto& key  = kernels[0]->get_key();
    auto& algo = key.algorithm;

    if(tile_m)
        *tile_m = algo.tile_shape.m;
    if(tile_n)
        *tile_n = algo.tile_shape.n;
    if(tile_k)
        *tile_k = algo.tile_shape.k;
    if(warp_tile_m)
        *warp_tile_m = algo.warp_tile_shape.m;
    if(warp_tile_n)
        *warp_tile_n = algo.warp_tile_shape.n;
    if(warp_tile_k)
        *warp_tile_k = algo.warp_tile_shape.k;
    if(warp_m)
        *warp_m = algo.wave_shape.m;
    if(warp_n)
        *warp_n = algo.wave_shape.n;
    if(warp_k)
        *warp_k = algo.wave_shape.k;

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
int dispatcher_select_kernel(int64_t M, int64_t N, int64_t K, char* name_buffer, int buffer_size)
{
    if(!g_initialized)
    {
        return -1;
    }

    Problem problem(M, N, K);
    auto kernel = g_dispatcher->select_kernel(problem);

    if(!kernel)
    {
        return -1;
    }

    std::string name = kernel->get_name();
    strncpy(name_buffer, name.c_str(), buffer_size - 1);
    name_buffer[buffer_size - 1] = '\0';

    return 0;
}

/**
 * Check if a problem size is supported by available kernels
 *
 * A problem is considered supported if at least one registered kernel
 * can handle the given dimensions. Support depends on:
 *
 * - Block tile divisibility: M, N, K must be divisible by the kernel's
 *   block tile sizes (TileM, TileN, TileK) unless padding is enabled
 * - Warp tile and wave configuration are internal to the kernel and
 *   affect performance but not dimension support
 *
 * For kernels with padding enabled (kPadM, kPadN, kPadK), any dimension
 * that has padding enabled does not require divisibility.
 *
 * Args:
 *   M, N, K: Problem dimensions (must be positive)
 *
 * Returns: 1 if supported, 0 if not supported or not initialized
 *
 * Example: For a kernel with TileM=128, TileN=128, TileK=32:
 *   - (1024, 1024, 512) -> supported (divisible)
 *   - (1000, 1024, 512) -> not supported (1000 % 128 != 0, unless padM enabled)
 */
int dispatcher_is_supported(int64_t M, int64_t N, int64_t K)
{
    if(!g_initialized)
    {
        return 0;
    }

    // Basic validation
    if(M <= 0 || N <= 0 || K <= 0)
    {
        return 0;
    }

    Problem problem(M, N, K);
    auto kernel = g_dispatcher->select_kernel(problem);
    return kernel != nullptr ? 1 : 0;
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
 * Returns: 0 on success, -1 on error, -2 if no kernel supports this size
 *
 * Note: This function:
 * 1. Allocates GPU memory
 * 2. Copies A, B to GPU
 * 3. Runs dispatcher GEMM
 * 4. Copies C back to CPU
 * 5. Frees GPU memory
 */
int dispatcher_run_gemm(const void* A, // Host pointer
                        const void* B, // Host pointer
                        void* C,       // Host pointer (output)
                        int64_t M,
                        int64_t N,
                        int64_t K,
                        float* time_ms) // Output
{
    if(!g_initialized || !A || !B || !C)
    {
        return -1;
    }

    // First check if any kernel supports this problem
    Problem problem(M, N, K);
    auto kernel = g_dispatcher->select_kernel(problem);
    if(!kernel)
    {
        // No kernel supports this problem size - return error code
        if(time_ms)
        {
            *time_ms = -1.0f;
        }
        return -2; // Special code for "no suitable kernel"
    }

    // Cast to correct types
    const ADataType* A_host = static_cast<const ADataType*>(A);
    const BDataType* B_host = static_cast<const BDataType*>(B);
    CDataType* C_host       = static_cast<CDataType*>(C);

    // Allocate GPU memory with proper cleanup on failure
    ADataType* A_dev = nullptr;
    BDataType* B_dev = nullptr;
    CDataType* C_dev = nullptr;

    // Helper lambda for cleanup
    auto cleanup_gpu_mem = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    if(hipMalloc(&A_dev, M * K * sizeof(ADataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMalloc(&B_dev, K * N * sizeof(BDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMalloc(&C_dev, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // Copy input data to GPU
    if(hipMemcpy(A_dev, A_host, M * K * sizeof(ADataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMemcpy(B_dev, B_host, K * N * sizeof(BDataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMemset(C_dev, 0, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // Run GEMM via dispatcher (kernel already selected, shouldn't throw)
    float exec_time;
    try
    {
        exec_time = g_dispatcher->run(A_dev, B_dev, C_dev, problem);
    }
    catch(const std::exception& e)
    {
        // Unexpected error during execution
        cleanup_gpu_mem();
        return -1;
    }

    // Copy result back to host
    if(hipMemcpy(C_host, C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // Store timing if requested
    if(time_ms)
    {
        *time_ms = exec_time;
    }

    // Cleanup GPU memory
    cleanup_gpu_mem();

    return 0;
}

/**
 * Get kernel information
 *
 * Returns: Pointer to null-terminated kernel name string
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Initialize dispatcher (alias for dispatcher_initialize)
 *
 * Returns: 0 on success, -1 on error
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Get the number of registered kernels
 *
 * Returns: Number of kernels in the registry
 */
int dispatcher_get_kernel_count() { return static_cast<int>(Registry::instance().size()); }

/**
 * Export registry to JSON string
 *
 * Returns: Pointer to static JSON string buffer (valid until next call)
 */
static std::string g_json_buffer;

const char* dispatcher_export_registry_json()
{
    auto& registry = Registry::instance();

    // Build JSON manually for simplicity
    std::ostringstream json;
    json << "{\n";
    json << "  \"metadata\": {\n";
    json << "    \"timestamp\": \"" << __DATE__ << " " << __TIME__ << "\",\n";
    json << "    \"total_kernels\": " << registry.size() << ",\n";
    json << "    \"export_version\": \"1.0\",\n";
    json << "    \"dispatcher_version\": \"1.0.0\"\n";
    json << "  },\n";
    json << "  \"statistics\": {\n";
    json << "    \"by_datatype\": {},\n";
    json << "    \"by_pipeline\": {},\n";
    json << "    \"by_scheduler\": {}\n";
    json << "  },\n";
    json << "  \"kernels\": [\n";

    auto kernels = registry.get_all();
    for(size_t i = 0; i < kernels.size(); ++i)
    {
        auto& kernel     = kernels[i];
        auto& key        = kernel->get_key();
        auto& algo       = key.algorithm;
        std::string name = kernel->get_name();

        json << "    {\n";
        json << "      \"identifier\": \"" << key.encode_identifier() << "\",\n";
        json << "      \"name\": \"" << name << "\",\n";
        json << "      \"algorithm\": {\n";
        json << "        \"tile_shape\": {\"m\": " << algo.tile_shape.m
             << ", \"n\": " << algo.tile_shape.n << ", \"k\": " << algo.tile_shape.k << "},\n";
        json << "        \"wave_shape\": {\"m\": " << unsigned(algo.wave_shape.m)
             << ", \"n\": " << unsigned(algo.wave_shape.n)
             << ", \"k\": " << unsigned(algo.wave_shape.k) << "},\n";
        json << "        \"warp_tile_shape\": {\"m\": " << unsigned(algo.warp_tile_shape.m)
             << ", \"n\": " << unsigned(algo.warp_tile_shape.n)
             << ", \"k\": " << unsigned(algo.warp_tile_shape.k) << "},\n";
        json << "        \"block_size\": " << algo.block_size << ",\n";
        json << "        \"persistent\": " << (algo.persistent ? "true" : "false") << ",\n";
        json << "        \"double_buffer\": " << (algo.double_buffer ? "true" : "false") << ",\n";
        json << "        \"preshuffle\": " << (algo.preshuffle ? "true" : "false") << ",\n";
        json << "        \"transpose_c\": " << (algo.transpose_c ? "true" : "false") << "\n";
        json << "      }\n";
        json << "    }";
        if(i < kernels.size() - 1)
        {
            json << ",";
        }
        json << "\n";
    }

    json << "  ]\n";
    json << "}\n";

    g_json_buffer = json.str();
    return g_json_buffer.c_str();
}

/**
 * Cleanup dispatcher resources
 */
void dispatcher_cleanup()
{
    // shared_ptr automatically handles cleanup when reset
    g_dispatcher.reset();
    g_initialized = false;
}

} // extern "C"
