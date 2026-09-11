// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace ck_tile {

/// @brief Scheduler for persistent GEMM kernels with asynchronous input streaming.
///
/// This structure enables signal-based synchronization for persistent kernels where input data
/// becomes available incrementally. It divides M-dimension tiles into chunks and uses signals
/// to coordinate between the input producer and the kernel consumer.
///
/// Uses modulo wraparound (like PyTorch's AsyncMM) for chunk index calculation:
///   chunk_idx = ((tile_idx + tile_idx_pivot_m) / tiles_per_chunk_m) % num_chunks
///
/// @par Typical usage pattern:
///   1. Set tiles_per_chunk_m to group tiles into chunks (e.g., 2 or 4 tiles per chunk)
///   2. Set tile_idx_pivot_m as offset for chunk calculation
///   3. Set num_chunks = ceil((tiles_m + tile_idx_pivot_m) / tiles_per_chunk_m)
///   4. Allocate chunk_signals array with size = num_chunks
///   5. Producer sets chunk_signals[i] = 1 when chunk i's data is ready
///   6. Kernel waits for chunk_signals[chunk_idx] before processing each tile
struct PersistentAsyncInputScheduler
{
    /// @brief Number of M-dimension tiles grouped into each chunk.
    /// Grouping tiles balances synchronization overhead against input streaming granularity.
    /// Set to 0 to disable async scheduling.
    uint32_t tiles_per_chunk_m = 0;

    /// @brief Device pointer to array of signal values (uint32_t), one per chunk.
    /// Producer sets signals to coordinate when input data is ready for consumption.
    /// Set to nullptr to disable async scheduling.
    uint32_t* chunk_signals = nullptr;

    /// @brief Pivot offset for rotating the chunk assignment.
    /// Allows shifting which tiles map to which chunks, useful for load balancing.
    /// chunk_idx = ((tile_idx + tile_idx_pivot_m) / tiles_per_chunk_m) % num_chunks
    int32_t tile_idx_pivot_m = 0;

    /// @brief Number of signal chunks allocated.
    /// Must equal ceil((tiles_m + tile_idx_pivot_m) / tiles_per_chunk_m).
    /// Modulo wraparound prevents out-of-bounds access when pivot shifts chunk assignment.
    uint32_t num_chunks = 0;
};

} // namespace ck_tile
