// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Heuristic Selection Example
 * 
 * Demonstrates how to use custom heuristic functions for kernel selection.
 * Shows how to select different kernels based on problem characteristics.
 */

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t err = call;                                             \
        if(err != hipSuccess) {                                            \
            std::cerr << "HIP error: " << hipGetErrorString(err) << "\n";  \
            exit(1);                                                        \
        }                                                                   \
    } while(0)

KernelKey create_kernel_key()
{
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
    key.signature.structured_sparsity = SelectedKernel::UseStructuredSparsity;
    
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

void run_gemm(Dispatcher& dispatcher, int M, int N, int K, const std::string& strategy_name)
{
    Problem problem(M, N, K);
    
    // Allocate GPU memory
    ADataType *a_dev, *b_dev;
    CDataType *c_dev;
    HIP_CHECK(hipMalloc(&a_dev, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&b_dev, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&c_dev, M * N * sizeof(CDataType)));
    
    // Initialize
    HIP_CHECK(hipMemset(a_dev, 1, M * K * sizeof(ADataType)));
    HIP_CHECK(hipMemset(b_dev, 1, K * N * sizeof(BDataType)));
    HIP_CHECK(hipMemset(c_dev, 0, M * N * sizeof(CDataType)));
    
    // Select kernel
    auto selected = dispatcher.select_kernel(problem);
    
    std::cout << "  Strategy: " << strategy_name << "\n";
    std::cout << "  Problem: " << M << "x" << N << "x" << K << "\n";
    
    if (selected) {
        std::cout << "  Selected: " << selected->get_name() << "\n";
        
        // Execute
        float time_ms = dispatcher.run(a_dev, b_dev, c_dev, problem, nullptr);
        float tflops = (2.0f * M * N * K) / (time_ms * 1e9);
        
        std::cout << "  Time: " << time_ms << " ms\n";
        std::cout << "  Performance: " << tflops << " TFLOPS\n";
    } else {
        std::cout << "  Selected: None (no matching kernel)\n";
    }
    
    // Cleanup
    HIP_CHECK(hipFree(a_dev));
    HIP_CHECK(hipFree(b_dev));
    HIP_CHECK(hipFree(c_dev));
}

int main(int argc, char** argv)
{
    std::cout << "======================================================================\n";
    std::cout << "CK Tile Dispatcher - Heuristic Selection Example\n";
    std::cout << "======================================================================\n\n";
    
    // GPU info
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (" << prop.gcnArchName << ")\n\n";
    
    // Register kernel
    auto key = create_kernel_key();
    auto kernel = create_generated_tile_kernel<
        SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(key, KERNEL_NAME);
    
    std::string kernel_id = key.encode_identifier();
    
    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Registry::Priority::High);
    
    std::cout << "Registered kernel: " << KERNEL_NAME << "\n";
    std::cout << "Kernel ID: " << kernel_id << "\n\n";
    
    // ==========================================================================
    // Demo 1: FirstFit Strategy (default)
    // ==========================================================================
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Demo 1: FirstFit Strategy (default)\n";
    std::cout << "----------------------------------------------------------------------\n";
    
    {
        Dispatcher dispatcher;
        dispatcher.set_strategy(Dispatcher::SelectionStrategy::FirstFit);
        
        run_gemm(dispatcher, 1024, 1024, 1024, "FirstFit");
    }
    std::cout << "\n";
    
    // ==========================================================================
    // Demo 2: Heuristic Strategy - Size-based selection
    // ==========================================================================
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Demo 2: Heuristic Strategy - Size-based selection\n";
    std::cout << "----------------------------------------------------------------------\n";
    
    {
        Dispatcher dispatcher;
        
        // Custom heuristic that prefers different kernels based on problem size
        dispatcher.set_heuristic([&kernel_id](const Problem& p) -> std::vector<std::string> {
            std::cout << "  [Heuristic called for " << p.M << "x" << p.N << "x" << p.K << "]\n";
            
            // For large problems (M*N > 1M), prefer larger tile sizes
            if (p.M * p.N >= 1024 * 1024) {
                std::cout << "  [Large problem - returning preferred kernels]\n";
            } else {
                std::cout << "  [Small problem - returning preferred kernels]\n";
            }
            
            // Return the kernel ID we have (in a real scenario, we'd return different IDs)
            return {kernel_id};
        });
        
        dispatcher.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
        
        // Small problem
        std::cout << "\nSmall problem:\n";
        run_gemm(dispatcher, 256, 256, 256, "Heuristic (size-based)");
        
        // Large problem
        std::cout << "\nLarge problem:\n";
        run_gemm(dispatcher, 2048, 2048, 2048, "Heuristic (size-based)");
    }
    std::cout << "\n";
    
    // ==========================================================================
    // Demo 3: Heuristic Strategy - Shape-aware selection
    // ==========================================================================
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Demo 3: Heuristic Strategy - Shape-aware selection\n";
    std::cout << "----------------------------------------------------------------------\n";
    
    {
        Dispatcher dispatcher;
        
        // Heuristic that considers matrix shape (tall, wide, square)
        dispatcher.set_heuristic([&kernel_id](const Problem& p) -> std::vector<std::string> {
            float aspect_ratio = static_cast<float>(p.M) / p.N;
            
            if (aspect_ratio > 2.0f) {
                std::cout << "  [Tall matrix (M >> N) - aspect ratio: " << aspect_ratio << "]\n";
            } else if (aspect_ratio < 0.5f) {
                std::cout << "  [Wide matrix (N >> M) - aspect ratio: " << aspect_ratio << "]\n";
            } else {
                std::cout << "  [Square-ish matrix - aspect ratio: " << aspect_ratio << "]\n";
            }
            
            // In a real scenario, return different kernel IDs based on shape
            return {kernel_id};
        });
        
        dispatcher.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
        
        // Square matrix
        std::cout << "\nSquare matrix:\n";
        run_gemm(dispatcher, 1024, 1024, 1024, "Heuristic (shape-aware)");
        
        // Tall matrix
        std::cout << "\nTall matrix:\n";
        run_gemm(dispatcher, 4096, 512, 1024, "Heuristic (shape-aware)");
        
        // Wide matrix
        std::cout << "\nWide matrix:\n";
        run_gemm(dispatcher, 512, 4096, 1024, "Heuristic (shape-aware)");
    }
    std::cout << "\n";
    
    // ==========================================================================
    // Demo 4: Dynamic strategy switching
    // ==========================================================================
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Demo 4: Dynamic strategy switching\n";
    std::cout << "----------------------------------------------------------------------\n";
    
    {
        Dispatcher dispatcher;
        
        // Start with FirstFit
        std::cout << "\nUsing FirstFit:\n";
        dispatcher.set_strategy(Dispatcher::SelectionStrategy::FirstFit);
        run_gemm(dispatcher, 1024, 1024, 1024, "FirstFit");
        
        // Switch to Heuristic
        std::cout << "\nSwitching to Heuristic:\n";
        dispatcher.set_heuristic([&kernel_id](const Problem& p) -> std::vector<std::string> {
            std::cout << "  [Heuristic invoked]\n";
            return {kernel_id};
        });
        dispatcher.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
        run_gemm(dispatcher, 1024, 1024, 1024, "Heuristic");
        
        // Switch back to FirstFit
        std::cout << "\nSwitching back to FirstFit:\n";
        dispatcher.set_strategy(Dispatcher::SelectionStrategy::FirstFit);
        run_gemm(dispatcher, 1024, 1024, 1024, "FirstFit");
    }
    
    std::cout << "\n======================================================================\n";
    std::cout << "Heuristic selection examples completed!\n";
    std::cout << "======================================================================\n";
    
    return 0;
}

