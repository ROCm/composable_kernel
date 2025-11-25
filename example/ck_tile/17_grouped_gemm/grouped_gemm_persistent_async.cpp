// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "run_grouped_gemm_example.inc"
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

    // TO-DO Add example

    if(!result)
        return -1;

    return 0;
}
