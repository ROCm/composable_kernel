// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file gemm_persistent_async.cpp
 * @brief Example demonstrating persistent GEMM with async input readiness
 *
 * This example shows how to use the PersistentAsyncScheduler for GEMM operations
 * where input data becomes ready asynchronously in chunks. This is particularly
 * useful in distributed computing scenarios where data arrives incrementally.
 *
 * Features demonstrated:
 * - Chunk-based async input signaling
 * - Producer-consumer synchronization
 * - Pivot-based tile traversal for hotspot spreading
 * - Persistent kernel execution
 */

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "gemm_persistent_async_invoker.hpp"
#include "persistent_async_scheduler.hpp"
#include "ck_tile/core/utility/gemm_validation.hpp"
#include <hip/hip_runtime.h>

/**
 * @brief Helper to allocate and initialize chunk signals
 *
 * @param num_chunks Number of chunks to allocate signals for
 * @param stream HIP stream for async operations
 * @return Device pointer to chunk signals array
 */
static uint32_t* allocate_chunk_signals(int num_chunks, hipStream_t stream)
{
    uint32_t* signals_device = nullptr;

    // Allocate device memory for signals
    ck_tile::hip_check_error(hipMalloc(&signals_device, num_chunks * sizeof(uint32_t)));

    // Initialize all signals to 0 (not ready)
    ck_tile::hip_check_error(
        hipMemsetAsync(signals_device, 0, num_chunks * sizeof(uint32_t), stream));

    return signals_device;
}

/**
 * @brief Helper to signal chunk readiness
 *
 * @param signals Device pointer to signals array
 * @param chunk_idx Index of chunk to signal
 * @param stream HIP stream for async operations
 */
static void signal_chunk_ready(uint32_t* signals, int chunk_idx, hipStream_t stream)
{
    uint32_t ready = 1;
    ck_tile::hip_check_error(hipMemcpyAsync(
        &signals[chunk_idx], &ready, sizeof(uint32_t), hipMemcpyHostToDevice, stream));
}

/**
 * @brief Simulate async data arrival by signaling chunks progressively
 *
 * In a real application, this would be triggered by actual data arrival events
 * (e.g., network communication, file I/O, etc.)
 */
static void
simulate_async_data_arrival(uint32_t* signals, int num_chunks, hipStream_t stream, int delay_ms = 1)
{
    // Signal chunks one by one with a small delay
    // In practice, this would be event-driven based on actual data availability
    for(int i = 0; i < num_chunks; ++i)
    {
        // Simulate delay in data arrival
        if(delay_ms > 0 && i > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }

        signal_chunk_ready(signals, i, stream);
    }
}

int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");
    std::string c_layout  = arg_parser.get_str("c_layout");

    std::tuple<ck_tile::index_t, ck_tile::index_t, ck_tile::index_t> gemm_sizes =
        parse_gemm_size(arg_parser);

    int m = std::get<0>(gemm_sizes);
    int n = std::get<1>(gemm_sizes);
    int k = std::get<2>(gemm_sizes);

    int stride_a = arg_parser.get_int("stride_a");
    int stride_b = arg_parser.get_int("stride_b");
    int stride_c = arg_parser.get_int("stride_c");

    // Async-specific parameters
    int tiles_per_chunk_m = arg_parser.get_int("tiles_per_chunk_m");
    int tile_idx_pivot_m  = arg_parser.get_int("tile_idx_pivot_m");
    bool enable_async     = arg_parser.get_int("enable_async") != 0;

    // using GemmConfig = GemmConfigMemoryInterwave;
    using DefaultGemmConfig = GemmConfigMemoryInterwave<ck_tile::half_t>;
    using Invoker           = PersistentAsyncInvoker;

    ck_tile::validate_gemm_stride(
        a_layout, b_layout, c_layout, m, n, k, stride_a, stride_b, stride_c);

    std::cout << "=== Persistent Async GEMM Example ===\n";
    std::cout << "Matrix dimensions: M=" << m << ", N=" << n << ", K=" << k << '\n';
    std::cout << "Async parameters:\n";
    std::cout << "  tiles_per_chunk_m: " << tiles_per_chunk_m << '\n';
    std::cout << "  tile_idx_pivot_m: " << tile_idx_pivot_m << '\n';
    std::cout << "  async_enabled: " << (enable_async ? "yes" : "no") << '\n';
    std::cout << "====================================\n\n";

    // Calculate number of chunks
    const int tiles_m    = (m + DefaultGemmConfig::M_Tile - 1) / DefaultGemmConfig::M_Tile;
    const int num_chunks = (tiles_m + tiles_per_chunk_m - 1) / tiles_per_chunk_m;

    if(tiles_m % tiles_per_chunk_m != 0)
    {
        std::cerr << "Warning: tiles_per_chunk_m (" << tiles_per_chunk_m
                  << ") does not evenly divide total M tiles (" << tiles_m << ")\n";
    }

    // Allocate and initialize chunk signals
    uint32_t* chunk_signals_device = nullptr;
    if(enable_async)
    {
        chunk_signals_device = allocate_chunk_signals(num_chunks, hipStreamDefault);
        std::cout << "Allocated " << num_chunks << " chunk signals\n";
    }

    // Create async args
    ck_tile::PersistentAsyncArgs async_args(
        enable_async ? tiles_per_chunk_m : 0, chunk_signals_device, tile_idx_pivot_m);

    // Launch async data arrival simulation in background thread if enabled
    std::thread data_arrival_thread;
    if(enable_async)
    {
        data_arrival_thread = std::thread([&]() {
            // Small delay before starting to simulate initial data latency
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            simulate_async_data_arrival(chunk_signals_device, num_chunks, hipStreamDefault, 5);
        });
    }

    int result = 0;
    try
    {
        if(data_type == "fp16")
        {
            using GemmConfig = GemmConfigMemoryInterwave<ck_tile::half_t>;
            result =
                run_gemm_example_prec_type_persistent_async<GemmConfig, Invoker, ck_tile::half_t>(
                    a_layout, b_layout, arg_parser, async_args);
        }
        else if(data_type == "bf16")
        {
            using GemmConfig = GemmConfigMemoryInterwave<ck_tile::bf16_t>;
            result =
                run_gemm_example_prec_type_persistent_async<GemmConfig, Invoker, ck_tile::bf16_t>(
                    a_layout, b_layout, arg_parser, async_args);
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        result = -1;
    }

    // Wait for data arrival thread to complete
    if(data_arrival_thread.joinable())
    {
        data_arrival_thread.join();
    }

    // Clean up
    if(chunk_signals_device)
    {
        ck_tile::hip_check_error(hipFree(chunk_signals_device));
    }

    return result;
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();

    // Add async-specific arguments
    arg_parser.insert(
        "tiles_per_chunk_m", "1", "Number of M tiles per chunk (granularity of async readiness)");
    arg_parser.insert(
        "tile_idx_pivot_m", "0", "Pivot offset for M dimension (for hotspot spreading)");
    arg_parser.insert("enable_async", "1", "Enable async input signaling (0=disabled, 1=enabled)");

    auto result = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {
        return !run_gemm_example(arg_parser);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
}
