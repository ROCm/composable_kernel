// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/// Complete C++ example demonstrating backend usage

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/backends/tile_backend.hpp"
#include "ck_tile/dispatcher/backends/library_backend.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

using namespace ck_tile::dispatcher;

/// Helper to allocate and initialize GPU memory
template <typename T>
T* allocate_device_memory(size_t size, bool initialize = true)
{
    T* ptr = nullptr;
    hipMalloc(&ptr, size * sizeof(T));
    
    if(initialize)
    {
        std::vector<T> host_data(size);
        for(size_t i = 0; i < size; ++i)
        {
            host_data[i] = static_cast<T>(rand()) / RAND_MAX;
        }
        hipMemcpy(ptr, host_data.data(), size * sizeof(T), hipMemcpyHostToDevice);
    }
    
    return ptr;
}

/// Example 1: Basic dispatcher usage with Tile backend
void example_tile_backend()
{
    std::cout << "=== Example 1: Tile Backend ===" << std::endl;
    
    // Create Tile backend
    backends::TileBackend backend;
    
    // Discover kernels from generated directory
    auto kernels = backend.discover_kernels("build/tile_engine/generated");
    
    std::cout << "Discovered " << kernels.size() << " tile kernels" << std::endl;
    
    // Register with registry
    auto& registry = Registry::instance();
    for(auto& kernel : kernels)
    {
        registry.register_kernel(kernel, Registry::Priority::High);
    }
    
    std::cout << "Registry size: " << registry.size() << std::endl;
}

/// Example 2: Library backend usage
void example_library_backend()
{
    std::cout << "\n=== Example 2: Library Backend ===" << std::endl;
    
    // Create Library backend
    backends::LibraryBackend backend;
    
    // Enumerate available operations
    auto operations = backend.enumerate_operations();
    std::cout << "Available operations:" << std::endl;
    for(const auto& op : operations)
    {
        std::cout << "  - " << op << std::endl;
    }
    
    // Discover library kernels
    auto kernels = backend.discover_kernels("");
    std::cout << "Discovered " << kernels.size() << " library kernels" << std::endl;
    
    // Register with registry
    auto& registry = Registry::instance();
    for(auto& kernel : kernels)
    {
        registry.register_kernel(kernel, Registry::Priority::Normal);
    }
}

/// Example 3: Mixed backend registration with conflict resolution
void example_mixed_backends()
{
    std::cout << "\n=== Example 3: Mixed Backends ===" << std::endl;
    
    auto& registry = Registry::instance();
    registry.clear();
    
    // Register Tile kernels (high priority)
    backends::TileBackend tile_backend;
    auto tile_kernels = tile_backend.discover_kernels("build/tile_engine/generated");
    
    for(auto& kernel : tile_kernels)
    {
        registry.register_kernel(kernel, Registry::Priority::High);
    }
    
    std::cout << "Registered " << tile_kernels.size() << " tile kernels (HIGH priority)" << std::endl;
    
    // Register Library kernels (normal priority)
    backends::LibraryBackend lib_backend;
    auto lib_kernels = lib_backend.discover_kernels("");
    
    for(auto& kernel : lib_kernels)
    {
        registry.register_kernel(kernel, Registry::Priority::Normal);
    }
    
    std::cout << "Registered " << lib_kernels.size() << " library kernels (NORMAL priority)" << std::endl;
    
    std::cout << "Total kernels in registry: " << registry.size() << std::endl;
    std::cout << "Note: Conflicts resolved in favor of Tile kernels (higher priority)" << std::endl;
}

/// Example 4: Kernel selection and execution
void example_kernel_execution()
{
    std::cout << "\n=== Example 4: Kernel Execution ===" << std::endl;
    
    // Setup problem
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    Problem problem;
    problem.M = M;
    problem.N = N;
    problem.K = K;
    problem.k_batch = 1;
    
    // Allocate device memory
    auto* a_ptr = allocate_device_memory<__half>(M * K);
    auto* b_ptr = allocate_device_memory<__half>(K * N);
    auto* c_ptr = allocate_device_memory<__half>(M * N, false);
    
    // Create dispatcher
    auto& registry = Registry::instance();
    Dispatcher dispatcher(&registry);
    
    // Select kernel
    auto kernel = dispatcher.select_kernel(problem);
    
    if(kernel)
    {
        std::cout << "Selected kernel: " << kernel->get_name() << std::endl;
        std::cout << "Backend type: " << 
            backends::KernelInstance::backend_type_to_string(kernel->get_backend_type()) << std::endl;
        
        // Execute kernel
        float time_ms = kernel->run(a_ptr, b_ptr, c_ptr, problem);
        
        std::cout << "Execution time: " << time_ms << " ms" << std::endl;
        
        // Calculate performance
        double flops = 2.0 * M * N * K;
        double gflops = flops / (time_ms * 1e6);
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    }
    else
    {
        std::cout << "No suitable kernel found for problem" << std::endl;
    }
    
    // Cleanup
    hipFree(a_ptr);
    hipFree(b_ptr);
    hipFree(c_ptr);
}

/// Example 5: Filtering kernels by criteria
void example_kernel_filtering()
{
    std::cout << "\n=== Example 5: Kernel Filtering ===" << std::endl;
    
    auto& registry = Registry::instance();
    
    // Filter by backend type
    auto tile_kernels = registry.filter([](const std::shared_ptr<backends::KernelInstance>& k) {
        return k->get_backend_type() == backends::BackendType::Tile;
    });
    
    std::cout << "Tile kernels: " << tile_kernels.size() << std::endl;
    
    // Filter by problem support
    Problem problem{.M = 2048, .N = 2048, .K = 2048};
    auto compatible_kernels = registry.filter([&problem](const std::shared_ptr<backends::KernelInstance>& k) {
        return k->supports(problem);
    });
    
    std::cout << "Kernels supporting 2048x2048x2048: " << compatible_kernels.size() << std::endl;
}

/// Example 6: Heuristic-based selection
void example_heuristic_selection()
{
    std::cout << "\n=== Example 6: Heuristic Selection ===" << std::endl;
    
    // Define a simple heuristic
    auto size_heuristic = [](const Problem& problem) -> std::vector<std::string> {
        int64_t total_size = problem.M * problem.N * problem.K;
        
        if(total_size < 1024 * 1024 * 1024)
        {
            // Small problem - prefer small tiles
            return {"gemm_128x128x32", "gemm_256x128x32"};
        }
        else
        {
            // Large problem - prefer large tiles
            return {"gemm_512x512x32", "gemm_256x256x32"};
        }
    };
    
    // Create dispatcher with heuristic
    auto& registry = Registry::instance();
    Dispatcher dispatcher(&registry);
    dispatcher.set_heuristic(size_heuristic);
    dispatcher.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
    
    // Test with different problem sizes
    std::vector<std::tuple<int, int, int>> problem_sizes = {
        {256, 256, 256},
        {2048, 2048, 2048},
        {4096, 4096, 4096}
    };
    
    for(const auto& [M, N, K] : problem_sizes)
    {
        Problem problem{.M = M, .N = N, .K = K};
        auto kernel = dispatcher.select_kernel(problem);
        
        if(kernel)
        {
            std::cout << "Problem " << M << "x" << N << "x" << K 
                     << " -> " << kernel->get_name() << std::endl;
        }
    }
}

int main()
{
    std::cout << "CK Tile Dispatcher - C++ Backend Examples" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try
    {
        example_tile_backend();
        example_library_backend();
        example_mixed_backends();
        example_kernel_execution();
        example_kernel_filtering();
        example_heuristic_selection();
        
        std::cout << "\n✓ All examples completed successfully" << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

