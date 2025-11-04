// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/// Example: How to integrate tile_engine generated kernels with the dispatcher

#include "ck_tile/dispatcher.hpp"

// Example: Include a tile_engine generated kernel header
// #include "tile_engine/gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x256x32_2x2x1_32x32x16.hpp"

namespace example {

using namespace ck_tile::dispatcher;

/// Step 1: Register tile_engine generated kernels
/// This would typically be done in an initialization function
void register_tile_kernels()
{
    auto& registry = Registry::instance();
    
    // Example: Register a kernel (uncomment when you have generated kernels)
    /*
    auto kernel = make_tile_kernel_instance<SelectedKernel>(
        DataType::FP16,      // dtype_a
        DataType::FP16,      // dtype_b
        DataType::FP16,      // dtype_c
        DataType::FP32,      // dtype_acc
        LayoutTag::RowMajor, // layout_a
        LayoutTag::ColMajor, // layout_b
        LayoutTag::RowMajor, // layout_c
        Pipeline::CompV4,
        Scheduler::Intrawave,
        Epilogue::CShuffle,
        942,                 // gfx942
        "gemm_fp16_rcr_compv4_cshuffle_intrawave_256x256x32_2x2x1_32x32x16"
    );
    
    registry.register_kernel(kernel, Registry::Priority::Normal);
    */
}

/// Step 2: Use the dispatcher for kernel selection and execution
void run_gemm_example(
    const void* a_ptr,
    const void* b_ptr,
    void* c_ptr,
    int M, int N, int K)
{
    // Create dispatcher
    Dispatcher dispatcher;
    
    // Define problem
    Problem problem(M, N, K);
    problem.prefer_persistent = false;
    problem.enable_validation = false;
    
    // Option 1: Automatic kernel selection
    try {
        float time = dispatcher.run(a_ptr, b_ptr, c_ptr, problem);
        printf("GEMM completed in %.3f ms\n", time);
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
    }
    
    // Option 2: Explicit kernel selection
    try {
        float time = dispatcher.run_explicit(
            "256x256x32_2x2x1_32x32x16_persist",
            a_ptr, b_ptr, c_ptr, nullptr, problem);
        printf("GEMM with explicit kernel completed in %.3f ms\n", time);
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
    }
}

/// Step 3: Query available kernels
void list_available_kernels()
{
    auto& registry = Registry::instance();
    
    auto all_kernels = registry.get_all();
    printf("Total registered kernels: %zu\n", all_kernels.size());
    
    for (const auto& kernel : all_kernels) {
        printf("  - %s\n", kernel->get_name().c_str());
    }
}

/// Step 4: Filter kernels by criteria
void find_persistent_kernels()
{
    auto& registry = Registry::instance();
    
    auto persistent_kernels = registry.filter([](const KernelInstance& k) {
        return k.get_key().algorithm.persistent;
    });
    
    printf("Found %zu persistent kernels\n", persistent_kernels.size());
}

/// Step 5: Use heuristics for kernel selection
void run_with_heuristics(
    const void* a_ptr,
    const void* b_ptr,
    void* c_ptr,
    int M, int N, int K)
{
    Dispatcher dispatcher;
    
    // Define a simple heuristic: prefer larger tile sizes for larger problems
    dispatcher.set_heuristic([](const Problem& problem) -> std::vector<std::string> {
        std::vector<std::string> candidates;
        
        if (problem.M >= 2048 && problem.N >= 2048) {
            // Large problem: prefer 256x256 tiles
            candidates.push_back("256x256x32_2x2x1_32x32x16_persist");
            candidates.push_back("256x256x64_2x2x1_32x32x16_persist");
        } else {
            // Smaller problem: prefer 128x128 tiles
            candidates.push_back("128x128x32_2x2x1_32x32x16_persist");
            candidates.push_back("128x128x64_2x2x1_32x32x16_persist");
        }
        
        return candidates;
    });
    
    Problem problem(M, N, K);
    float time = dispatcher.run(a_ptr, b_ptr, c_ptr, problem);
    printf("GEMM with heuristics completed in %.3f ms\n", time);
}

} // namespace example

/// Main function showing typical usage pattern
int main()
{
    // Initialize: Register all available kernels
    example::register_tile_kernels();
    
    // List what's available
    example::list_available_kernels();
    
    // Find specific kernel types
    example::find_persistent_kernels();
    
    // Example usage would go here
    // example::run_gemm_example(a_ptr, b_ptr, c_ptr, 1024, 1024, 1024);
    
    printf("Dispatcher example completed\n");
    return 0;
}

