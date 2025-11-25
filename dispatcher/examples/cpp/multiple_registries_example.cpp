// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example: Multiple Registries
 * 
 * Demonstrates how to use multiple independent registries with dispatchers.
 * This is useful for:
 * - Organizing kernels by data type (FP16, BF16, FP32)
 * - Separating kernels by operation type (GEMM, Conv, Attention)
 * - Having different kernel sets for different use cases
 * 
 * Usage:
 *   ./multiple_registries_example
 */

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/json_export.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>

// The generated kernel header is included via -include compiler flag
using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;

// Helper to check HIP errors
#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t err = call;                                             \
        if(err != hipSuccess) {                                            \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__    \
                      << ": " << hipGetErrorString(err) << std::endl;      \
            exit(1);                                                        \
        }                                                                   \
    } while(0)

KernelKey create_kernel_key()
{
    KernelKey key;
    
    // Signature
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
    key.signature.structured_sparsity = SelectedKernel::UseStructuredSparsity;
    
    // Algorithm - extract from SelectedKernel
    key.algorithm.tile_shape.m = SelectedKernel::TileM;
    key.algorithm.tile_shape.n = SelectedKernel::TileN;
    key.algorithm.tile_shape.k = SelectedKernel::TileK;
    key.algorithm.wave_shape.m = SelectedKernel::WarpPerBlock_M;
    key.algorithm.wave_shape.n = SelectedKernel::WarpPerBlock_N;
    key.algorithm.wave_shape.k = SelectedKernel::WarpPerBlock_K;
    key.algorithm.warp_tile_shape.m = SelectedKernel::WarpTileM;
    key.algorithm.warp_tile_shape.n = SelectedKernel::WarpTileN;
    key.algorithm.warp_tile_shape.k = SelectedKernel::WarpTileK;
    key.algorithm.pipeline = Pipeline::CompV4;
    key.algorithm.scheduler = Scheduler::Intrawave;
    key.algorithm.epilogue = Epilogue::CShuffle;
    key.algorithm.block_size = SelectedKernel::BlockSize;
    key.algorithm.double_buffer = SelectedKernel::DoubleSmemBuffer;
    key.algorithm.persistent = SelectedKernel::UsePersistentKernel;
    key.algorithm.preshuffle = SelectedKernel::Preshuffle;
    key.algorithm.transpose_c = SelectedKernel::TransposeC;
    key.algorithm.num_wave_groups = SelectedKernel::NumWaveGroups;
    key.gfx_arch = "gfx942";
    
    return key;
}

int main(int argc, char** argv)
{
    std::cout << "======================================================================\n";
    std::cout << "CK Tile Dispatcher - Multiple Registries Example\n";
    std::cout << "======================================================================\n\n";
    
    // GPU info
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    
    if(device_count == 0) {
        std::cerr << "No HIP devices found!\n";
        return 1;
    }
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (" << prop.gcnArchName << ")\n\n";
    
    // Create the kernel instance
    auto key = create_kernel_key();
    auto kernel = create_generated_tile_kernel<
        SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
        key, std::string(KERNEL_NAME));
    
    // ============================================================
    // Method 1: Multiple standalone registries
    // ============================================================
    std::cout << "=== Method 1: Multiple Standalone Registries ===\n\n";
    
    // Create separate registries
    Registry fp16_registry;
    fp16_registry.set_name("fp16_gemm_kernels");
    
    Registry production_registry;
    production_registry.set_name("production_kernels");
    
    Registry experimental_registry;
    experimental_registry.set_name("experimental_kernels");
    
    // Register the kernel to different registries
    fp16_registry.register_kernel(kernel, Registry::Priority::High);
    production_registry.register_kernel(kernel, Registry::Priority::Normal);
    experimental_registry.register_kernel(kernel, Registry::Priority::Low);
    
    std::cout << "Created 3 registries:\n";
    std::cout << "  - " << fp16_registry.get_name() << ": " << fp16_registry.size() << " kernel(s)\n";
    std::cout << "  - " << production_registry.get_name() << ": " << production_registry.size() << " kernel(s)\n";
    std::cout << "  - " << experimental_registry.get_name() << ": " << experimental_registry.size() << " kernel(s)\n\n";
    
    // ============================================================
    // Method 2: Create dispatchers with specific registries
    // ============================================================
    std::cout << "=== Method 2: Dispatchers with Specific Registries ===\n\n";
    
    // Create dispatchers pointing to different registries
    Dispatcher fp16_dispatcher(&fp16_registry);
    Dispatcher production_dispatcher(&production_registry);
    Dispatcher experimental_dispatcher(&experimental_registry);
    
    std::cout << "Created 3 dispatchers, each using a different registry\n\n";
    
    // ============================================================
    // Method 3: Select kernels from different registries
    // ============================================================
    std::cout << "=== Method 3: Kernel Selection from Different Registries ===\n\n";
    
    Problem problem(1024, 1024, 1024);
    
    auto k1 = fp16_dispatcher.select_kernel(problem);
    auto k2 = production_dispatcher.select_kernel(problem);
    auto k3 = experimental_dispatcher.select_kernel(problem);
    
    std::cout << "Kernel selection for problem M=1024, N=1024, K=1024:\n";
    std::cout << "  - From fp16_registry: " << (k1 ? k1->get_name() : "none") << "\n";
    std::cout << "  - From production_registry: " << (k2 ? k2->get_name() : "none") << "\n";
    std::cout << "  - From experimental_registry: " << (k3 ? k3->get_name() : "none") << "\n\n";
    
    // ============================================================
    // Method 4: Merge registries
    // ============================================================
    std::cout << "=== Method 4: Merge Registries ===\n\n";
    
    Registry combined_registry;
    combined_registry.set_name("combined_kernels");
    
    // Merge from other registries
    auto merged_from_fp16 = combined_registry.merge_from(fp16_registry, Registry::Priority::High);
    auto merged_from_exp = combined_registry.merge_from(experimental_registry, Registry::Priority::Low);
    
    std::cout << "Created combined registry by merging:\n";
    std::cout << "  - Merged " << merged_from_fp16 << " kernel(s) from fp16_registry\n";
    std::cout << "  - Merged " << merged_from_exp << " kernel(s) from experimental_registry\n";
    std::cout << "  - Combined total: " << combined_registry.size() << " kernel(s)\n\n";
    
    // ============================================================
    // Method 5: Auto-export each registry to separate JSON files
    // ============================================================
    std::cout << "=== Method 5: Auto-Export to Separate JSON Files ===\n\n";
    
    fp16_registry.enable_auto_export("fp16_kernels.json", true, false);
    production_registry.enable_auto_export("production_kernels.json", true, false);
    combined_registry.enable_auto_export("combined_kernels.json", true, false);
    
    std::cout << "Auto-export enabled for:\n";
    std::cout << "  - fp16_registry -> fp16_kernels.json\n";
    std::cout << "  - production_registry -> production_kernels.json\n";
    std::cout << "  - combined_registry -> combined_kernels.json\n\n";
    
    // ============================================================
    // Method 6: Using the factory function
    // ============================================================
    std::cout << "=== Method 6: Using Factory Function ===\n\n";
    
    auto custom_registry = make_registry("my_custom_kernels");
    custom_registry->register_kernel(kernel, Registry::Priority::Normal);
    
    std::cout << "Created registry via make_registry():\n";
    std::cout << "  - Name: " << custom_registry->get_name() << "\n";
    std::cout << "  - Kernels: " << custom_registry->size() << "\n\n";
    
    // ============================================================
    // Method 7: Global singleton (backward compatible)
    // ============================================================
    std::cout << "=== Method 7: Global Singleton (Backward Compatible) ===\n\n";
    
    Registry::instance().clear();
    Registry::instance().set_name("global_singleton");
    Registry::instance().register_kernel(kernel, Registry::Priority::High);
    
    // Default dispatcher uses the singleton
    Dispatcher default_dispatcher;
    auto k_default = default_dispatcher.select_kernel(problem);
    
    std::cout << "Global singleton registry:\n";
    std::cout << "  - Name: " << Registry::instance().get_name() << "\n";
    std::cout << "  - Kernels: " << Registry::instance().size() << "\n";
    std::cout << "  - Default dispatcher selects: " << (k_default ? k_default->get_name() : "none") << "\n\n";
    
    // ============================================================
    // Execute GEMM using a specific registry's dispatcher
    // ============================================================
    std::cout << "=== Execute GEMM Using FP16 Registry ===\n\n";
    
    int M = 1024, N = 1024, K = 1024;
    
    // Allocate GPU memory
    ADataType *a_dev, *b_dev;
    CDataType *c_dev;
    HIP_CHECK(hipMalloc(&a_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&b_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&c_dev, M * N * sizeof(CDataType)));
    
    // Initialize with random data
    std::vector<ADataType> a_host(M * K);
    std::vector<BDataType> b_host(K * N);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : a_host) val = ADataType(dis(gen));
    for (auto& val : b_host) val = BDataType(dis(gen));
    
    HIP_CHECK(hipMemcpy(a_dev, a_host.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_dev, b_host.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(c_dev, 0, M * N * sizeof(CDataType)));
    
    // Execute via the FP16 dispatcher (using fp16_registry)
    Problem exec_problem(M, N, K);
    float time_ms = fp16_dispatcher.run(a_dev, b_dev, c_dev, exec_problem, nullptr);
    
    // Calculate performance
    float tflops = (2.0f * M * N * K) / (time_ms * 1e9);
    
    std::cout << "Executed GEMM " << M << "x" << N << "x" << K << " via fp16_dispatcher:\n";
    std::cout << "  Time: " << time_ms << " ms\n";
    std::cout << "  Performance: " << tflops << " TFLOPS\n\n";
    
    // Cleanup
    HIP_CHECK(hipFree(a_dev));
    HIP_CHECK(hipFree(b_dev));
    HIP_CHECK(hipFree(c_dev));
    
    std::cout << "======================================================================\n";
    std::cout << "Multiple Registries Example Complete!\n";
    std::cout << "======================================================================\n\n";
    
    std::cout << "JSON files will be created on exit:\n";
    std::cout << "  - fp16_kernels.json\n";
    std::cout << "  - production_kernels.json\n";
    std::cout << "  - combined_kernels.json\n";
    
    return 0;
}

