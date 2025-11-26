// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 05: Heuristics
 *
 * Demonstrates kernel selection strategies: FirstFit and custom heuristics.
 *
 * Complexity: ★★★★☆
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

// Custom heuristic: returns ranked list of kernel identifiers based on problem size
std::vector<std::string> size_based_heuristic(const Problem& problem)
{
    // Return kernel identifiers ranked by preference
    // For larger problems, prefer larger tile kernels
    if(problem.M >= 2048 && problem.N >= 2048)
    {
        return {KERNEL_NAME}; // Use the available kernel
    }
    else
    {
        return {KERNEL_NAME}; // Same kernel (we only have one)
    }
}

int main()
{
    print_header("Example 05: Heuristics");

    // Setup kernel
    KernelKeyBuilder builder = KernelKeyBuilder::fp16_rcr();
    builder.tile_m           = SelectedKernel::TileM;
    builder.tile_n           = SelectedKernel::TileN;
    builder.tile_k           = SelectedKernel::TileK;
    builder.wave_m           = SelectedKernel::WarpPerBlock_M;
    builder.wave_n           = SelectedKernel::WarpPerBlock_N;
    builder.wave_k           = SelectedKernel::WarpPerBlock_K;
    builder.warp_m           = SelectedKernel::WarpTileM;
    builder.warp_n           = SelectedKernel::WarpTileN;
    builder.warp_k           = SelectedKernel::WarpTileK;
    builder.block_size       = SelectedKernel::BlockSize;

    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            builder.build(), KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel);

    std::cout << "Registered kernel: " << KERNEL_NAME << "\n\n";

    std::vector<std::tuple<int, int, int>> sizes = {
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
    };

    // Demo 1: FirstFit Strategy
    std::cout << "Demo 1: FirstFit Strategy\n";
    std::cout << "  Uses first kernel that supports the problem\n";
    print_separator('-', 50);

    Dispatcher dispatcher_ff;
    dispatcher_ff.set_strategy(Dispatcher::SelectionStrategy::FirstFit);

    for(const auto& [M, N, K] : sizes)
    {
        Problem problem(M, N, K);

        GpuBuffer<ADataType> a_dev(M * K);
        GpuBuffer<BDataType> b_dev(K * N);
        GpuBuffer<CDataType> c_dev(M * N);

        std::vector<ADataType> a_host(M * K, ADataType(1.0f));
        std::vector<BDataType> b_host(K * N, BDataType(1.0f));

        a_dev.copy_from_host(a_host.data());
        b_dev.copy_from_host(b_host.data());
        c_dev.zero();

        float t       = dispatcher_ff.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
        double tflops = calculate_tflops(M, N, K, t);
        std::cout << "  " << format_size(M, N, K) << ": " << std::fixed << std::setprecision(4) << t
                  << " ms (" << std::setprecision(2) << tflops << " TFLOPS)\n";
    }

    // Demo 2: Heuristic Strategy with custom function
    std::cout << "\nDemo 2: Heuristic Strategy\n";
    std::cout << "  Uses custom heuristic to rank kernels\n";
    print_separator('-', 50);

    Dispatcher dispatcher_heur;
    dispatcher_heur.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
    dispatcher_heur.set_heuristic(size_based_heuristic);

    for(const auto& [M, N, K] : sizes)
    {
        Problem problem(M, N, K);

        GpuBuffer<ADataType> a_dev(M * K);
        GpuBuffer<BDataType> b_dev(K * N);
        GpuBuffer<CDataType> c_dev(M * N);

        std::vector<ADataType> a_host(M * K, ADataType(1.0f));
        std::vector<BDataType> b_host(K * N, BDataType(1.0f));

        a_dev.copy_from_host(a_host.data());
        b_dev.copy_from_host(b_host.data());
        c_dev.zero();

        float t = dispatcher_heur.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
        double tflops = calculate_tflops(M, N, K, t);
        std::cout << "  " << format_size(M, N, K) << ": " << std::fixed << std::setprecision(4) << t
                  << " ms (" << std::setprecision(2) << tflops << " TFLOPS)\n";
    }

    // Demo 3: Show selection without execution
    std::cout << "\nDemo 3: Kernel Selection\n";
    print_separator('-', 50);

    Dispatcher dispatcher;
    for(const auto& [M, N, K] : sizes)
    {
        Problem problem(M, N, K);
        auto selected = dispatcher.select_kernel(problem);
        std::cout << "  " << format_size(M, N, K) << " -> ";
        if(selected)
        {
            std::cout << selected->get_name() << "\n";
        }
        else
        {
            std::cout << "(no kernel found)\n";
        }
    }

    print_separator();
    std::cout << "Heuristics demo complete!\n";
    print_separator();

    return 0;
}
