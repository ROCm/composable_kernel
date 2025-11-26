// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 09: Multiple Registries with Different Kernels
 *
 * Demonstrates registering different kernel configurations to different registries.
 * Each registry can have kernels optimized for different use cases:
 *   - compute_registry: compute-bound optimized (larger tiles)
 *   - memory_registry: memory-bound optimized (smaller tiles)
 *   - latency_registry: low-latency optimized (smallest tiles)
 *
 * In production, each registry would have kernels generated with different
 * configurations. This example shows the pattern using the same underlying
 * kernel but with different key configurations.
 *
 * Complexity: ★★★★★
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

// Helper to create kernel with custom configuration
KernelInstancePtr create_kernel_with_config(int tile_m,
                                            int tile_n,
                                            int tile_k,
                                            const std::string& name,
                                            Pipeline pipeline = Pipeline::CompV4)
{
    KernelKeyBuilder builder = KernelKeyBuilder::fp16_rcr();

    // Custom tile configuration
    builder.tile_m = tile_m;
    builder.tile_n = tile_n;
    builder.tile_k = tile_k;

    // Use actual kernel's wave/warp config
    builder.wave_m     = SelectedKernel::WarpPerBlock_M;
    builder.wave_n     = SelectedKernel::WarpPerBlock_N;
    builder.wave_k     = SelectedKernel::WarpPerBlock_K;
    builder.warp_m     = SelectedKernel::WarpTileM;
    builder.warp_n     = SelectedKernel::WarpTileN;
    builder.warp_k     = SelectedKernel::WarpTileK;
    builder.block_size = SelectedKernel::BlockSize;
    builder.pipeline   = pipeline;

    return create_generated_tile_kernel<SelectedKernel,
                                        ADataType,
                                        BDataType,
                                        CDataType,
                                        AccDataType>(builder.build(), name);
}

int main()
{
    print_header("Example 09: Multiple Registries with Different Kernels");

    // =========================================================================
    // Part 1: Create registries for different optimization targets
    // =========================================================================
    std::cout << "Part 1: Create specialized registries\n";
    print_separator('-', 60);

    // Registry for compute-bound workloads (large matrices)
    Registry compute_registry;
    compute_registry.set_name("compute_optimized");

    // Registry for memory-bound workloads (bandwidth limited)
    Registry memory_registry;
    memory_registry.set_name("memory_optimized");

    // Registry for latency-sensitive workloads (small matrices)
    Registry latency_registry;
    latency_registry.set_name("latency_optimized");

    std::cout << "  compute_registry: for large matrices (compute-bound)\n";
    std::cout << "  memory_registry:  for medium matrices (bandwidth-limited)\n";
    std::cout << "  latency_registry: for small matrices (latency-sensitive)\n\n";

    // =========================================================================
    // Part 2: Register different kernel configs to each registry
    // =========================================================================
    std::cout << "Part 2: Register different kernels to each registry\n";
    print_separator('-', 60);

    // Compute-optimized: larger tiles for better compute efficiency
    // In production: generate kernels with --tile 256x256x64
    auto compute_kernel_1 =
        create_kernel_with_config(256, 256, 64, "compute_256x256x64", Pipeline::CompV4);
    auto compute_kernel_2 =
        create_kernel_with_config(256, 128, 64, "compute_256x128x64", Pipeline::CompV4);

    compute_registry.register_kernel(compute_kernel_1, Registry::Priority::High);
    compute_registry.register_kernel(compute_kernel_2, Registry::Priority::Normal);
    std::cout << "  compute_registry: added 2 large-tile kernels\n";

    // Memory-optimized: medium tiles with memory-focused pipeline
    // In production: generate kernels with --pipeline memory
    auto memory_kernel_1 =
        create_kernel_with_config(128, 128, 32, "memory_128x128x32", Pipeline::CompV3);
    auto memory_kernel_2 =
        create_kernel_with_config(128, 64, 32, "memory_128x64x32", Pipeline::CompV3);
    auto memory_kernel_3 =
        create_kernel_with_config(64, 128, 32, "memory_64x128x32", Pipeline::CompV3);

    memory_registry.register_kernel(memory_kernel_1, Registry::Priority::High);
    memory_registry.register_kernel(memory_kernel_2, Registry::Priority::Normal);
    memory_registry.register_kernel(memory_kernel_3, Registry::Priority::Normal);
    std::cout << "  memory_registry:  added 3 medium-tile kernels\n";

    // Latency-optimized: smallest tiles for quick execution
    // In production: generate kernels with --tile 64x64x32 or smaller
    auto latency_kernel_1 =
        create_kernel_with_config(64, 64, 32, "latency_64x64x32", Pipeline::CompV4);
    auto latency_kernel_2 =
        create_kernel_with_config(32, 64, 32, "latency_32x64x32", Pipeline::CompV4);
    auto latency_kernel_3 =
        create_kernel_with_config(64, 32, 32, "latency_64x32x32", Pipeline::CompV4);
    auto latency_kernel_4 =
        create_kernel_with_config(32, 32, 32, "latency_32x32x32", Pipeline::CompV4);

    latency_registry.register_kernel(latency_kernel_1, Registry::Priority::High);
    latency_registry.register_kernel(latency_kernel_2, Registry::Priority::Normal);
    latency_registry.register_kernel(latency_kernel_3, Registry::Priority::Normal);
    latency_registry.register_kernel(latency_kernel_4, Registry::Priority::Low);
    std::cout << "  latency_registry: added 4 small-tile kernels\n\n";

    // =========================================================================
    // Part 3: Show registry contents
    // =========================================================================
    std::cout << "Part 3: Registry contents\n";
    print_separator('-', 60);

    std::cout << "  compute_registry: " << compute_registry.size() << " kernels\n";
    std::cout << "  memory_registry:  " << memory_registry.size() << " kernels\n";
    std::cout << "  latency_registry: " << latency_registry.size() << " kernels\n\n";

    // =========================================================================
    // Part 4: Create dispatchers and select kernels
    // =========================================================================
    std::cout << "Part 4: Kernel selection for different problem sizes\n";
    print_separator('-', 60);

    Dispatcher compute_dispatcher(&compute_registry);
    Dispatcher memory_dispatcher(&memory_registry);
    Dispatcher latency_dispatcher(&latency_registry);

    // Show which kernel each dispatcher would select for different sizes
    std::vector<std::tuple<int, int, int, const char*>> test_cases = {
        {4096, 4096, 4096, "Large (compute-bound)"},
        {1024, 1024, 1024, "Medium (balanced)"},
        {256, 256, 256, "Small (latency-sensitive)"},
    };

    for(const auto& [M, N, K, desc] : test_cases)
    {
        Problem problem(M, N, K);

        auto compute_kernel = compute_dispatcher.select_kernel(problem);
        auto memory_kernel  = memory_dispatcher.select_kernel(problem);
        auto latency_kernel = latency_dispatcher.select_kernel(problem);

        std::cout << "  " << desc << " (" << M << "x" << N << "x" << K << "):\n";
        if(compute_kernel)
            std::cout << "    compute: " << compute_kernel->get_name() << "\n";
        if(memory_kernel)
            std::cout << "    memory:  " << memory_kernel->get_name() << "\n";
        if(latency_kernel)
            std::cout << "    latency: " << latency_kernel->get_name() << "\n";
        std::cout << "\n";
    }

    // =========================================================================
    // Part 5: Execute with each dispatcher
    // =========================================================================
    std::cout << "Part 5: Execute GEMM with each dispatcher\n";
    print_separator('-', 60);

    const int M = 1024, N = 1024, K = 1024;
    Problem problem(M, N, K);

    GpuBuffer<ADataType> a_dev(M * K);
    GpuBuffer<BDataType> b_dev(K * N);
    GpuBuffer<CDataType> c_dev(M * N);

    std::vector<ADataType> a_host(M * K, ADataType(1.0f));
    std::vector<BDataType> b_host(K * N, BDataType(1.0f));

    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());

    std::cout << "  Problem size: " << format_size(M, N, K) << "\n\n";

    c_dev.zero();
    float compute_time =
        compute_dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
    std::cout << "  compute_dispatcher: " << std::fixed << std::setprecision(4) << compute_time
              << " ms (" << std::setprecision(2) << calculate_tflops(M, N, K, compute_time)
              << " TFLOPS)\n";

    c_dev.zero();
    float memory_time =
        memory_dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
    std::cout << "  memory_dispatcher:  " << std::setprecision(4) << memory_time << " ms ("
              << std::setprecision(2) << calculate_tflops(M, N, K, memory_time) << " TFLOPS)\n";

    c_dev.zero();
    float latency_time =
        latency_dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
    std::cout << "  latency_dispatcher: " << std::setprecision(4) << latency_time << " ms ("
              << std::setprecision(2) << calculate_tflops(M, N, K, latency_time) << " TFLOPS)\n\n";

    // =========================================================================
    // Part 6: Merge all registries into one
    // =========================================================================
    std::cout << "Part 6: Merge all registries\n";
    print_separator('-', 60);

    Registry unified_registry;
    unified_registry.set_name("unified");

    unified_registry.merge_from(compute_registry, Registry::Priority::High);
    unified_registry.merge_from(memory_registry, Registry::Priority::Normal);
    unified_registry.merge_from(latency_registry, Registry::Priority::Low);

    std::cout << "  Merged all registries into unified_registry\n";
    std::cout << "  Total kernels: " << unified_registry.size() << "\n\n";

    // =========================================================================
    // Part 7: Export each registry to JSON
    // =========================================================================
    std::cout << "Part 7: Export to JSON\n";
    print_separator('-', 60);

    std::cout << "  compute_registry: " << compute_registry.export_json().length() << " bytes\n";
    std::cout << "  memory_registry:  " << memory_registry.export_json().length() << " bytes\n";
    std::cout << "  latency_registry: " << latency_registry.export_json().length() << " bytes\n";
    std::cout << "  unified_registry: " << unified_registry.export_json().length() << " bytes\n\n";

    print_separator();
    std::cout << "Example 09 complete!\n";
    std::cout << "\nNote: In production, generate actual different kernels:\n";
    std::cout << "  python3 unified_gemm_codegen.py --tile 256x256x64  # compute\n";
    std::cout << "  python3 unified_gemm_codegen.py --tile 128x128x32  # memory\n";
    std::cout << "  python3 unified_gemm_codegen.py --tile 64x64x32    # latency\n";
    print_separator();

    return 0;
}
