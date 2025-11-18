// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file persistent_async_scheduler.hpp
 * @brief HIP-based Persistent Async Input Scheduler for CK Tile GEMM
 *
 * This file implements a persistent async input scheduler similar to CUTLASS's
 * PersistentAsyncInputScheduler, adapted for AMD GPUs using HIP.
 *
 * Features:
 * - tiles_per_chunk_m: Granularity at which data becomes ready
 * - chunk_signals: Global memory flags indicating chunk readiness
 * - tile_idx_pivot_m: Post-swizzle pivot to spread hotspots
 * - Producer-consumer wait mechanism for async data readiness
 */

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

/**
 * @brief Wait for signal to become ready using HIP atomic operations
 *
 * @param addr Pointer to signal in global memory
 *
 * This function implements a busy-wait on a global memory flag using
 * volatile loads with acquire semantics for AMD GPUs.
 */
__device__ void wait_signal(uint32_t* addr)
{
    uint32_t ready = __atomic_load_n(addr, __ATOMIC_ACQUIRE);
    while(!ready)
    {
        // Use volatile load to prevent compiler optimization
        asm volatile("flat_load_dword %0, %1 glc\n"
                     "s_waitcnt vmcnt(0)"
                     : "=v"(ready)
                     : "v"(addr)
                     : "memory");

        // Add a small delay to reduce memory traffic
        __builtin_amdgcn_s_sleep(1);

        ready = __atomic_load_n(addr, __ATOMIC_ACQUIRE);
    }
}

/**
 * @brief Iteration boundary fence for persistent kernels
 *
 * Ensures all memory operations complete before moving to next tile.
 * Required for safe LDS reuse in persistent loops.
 */
__device__ void iteration_boundary_fence()
{
    // Wait for all vector memory operations
    __builtin_amdgcn_s_waitcnt(/*vmcnt*/ 0 | (/*lgkmcnt*/ 0 << 8));
    // Workgroup barrier
    __builtin_amdgcn_s_barrier();
}

/**
 * @brief Arguments for persistent async input scheduler
 *
 * This structure extends the standard GEMM arguments with async-specific
 * parameters for controlling data readiness signaling and tile traversal.
 */
struct PersistentAsyncArgs
{
    /// @brief Number of M tiles in each chunk (granularity of data readiness)
    index_t tiles_per_chunk_m = 0;

    /// @brief Pointer to chunk readiness signals in global memory
    /// chunk_signals[i] == 1 indicates chunk i is ready for processing
    uint32_t* chunk_signals = nullptr;

    /// @brief Pivot offset for M dimension after swizzling
    /// Allows different ranks to process different M tiles simultaneously
    index_t tile_idx_pivot_m = 0;

    CK_TILE_HOST PersistentAsyncArgs() = default;

    CK_TILE_HOST PersistentAsyncArgs(index_t tiles_per_chunk_m_,
                                     uint32_t* chunk_signals_,
                                     index_t tile_idx_pivot_m_)
        : tiles_per_chunk_m(tiles_per_chunk_m_),
          chunk_signals(chunk_signals_),
          tile_idx_pivot_m(tile_idx_pivot_m_)
    {
    }
};

/**
 * @brief Persistent Async Tile Scheduler Implementation
 *
 * This scheduler manages work distribution for persistent GEMM kernels
 * with asynchronous input readiness. It extends the basic persistent
 * scheduler with:
 * - Chunk-based data readiness signaling
 * - Producer-consumer synchronization
 * - Pivot-based tile traversal for hotspot spreading
 *
 * @tparam TilePartitioner_ The tile partitioner type
 */
template <typename TilePartitioner_>
struct PersistentAsyncScheduler
{
    using TilePartitioner = remove_cvref_t<TilePartitioner_>;

    struct WorkTileInfo
    {
        index_t tile_idx_m;
        index_t tile_idx_n;
        index_t batch_idx;
        bool is_valid;

        CK_TILE_DEVICE bool IsValid() const { return is_valid; }

        CK_TILE_DEVICE static WorkTileInfo InvalidTile() { return WorkTileInfo{-1, -1, -1, false}; }
    };

    struct SchedulerState
    {
        index_t current_tile_linear;
        index_t total_tiles_m;
        index_t total_tiles_n;
        index_t total_grid_size;
        bool is_mainloop_producer;

        // Async-specific state
        index_t tiles_per_chunk_m;
        uint32_t* chunk_signals;
        index_t tile_idx_pivot_m;
    };

    CK_TILE_DEVICE PersistentAsyncScheduler(const PersistentAsyncArgs& async_args,
                                            index_t tiles_m,
                                            index_t tiles_n,
                                            index_t grid_size)
    {
        state_.current_tile_linear = blockIdx.x + blockIdx.y * gridDim.x;
        state_.total_tiles_m       = tiles_m;
        state_.total_tiles_n       = tiles_n;
        state_.total_grid_size     = grid_size;

        // Determine if this wave is a mainloop producer
        // Only the first wave in the first wave group is the producer
        const index_t warp_id       = threadIdx.x / warpSize;
        const index_t wave_group_id = warp_id / 4; // 4 waves per wave group
        state_.is_mainloop_producer = (wave_group_id == 0) && (warp_id % 4 == 0);

        // Async parameters
        state_.tiles_per_chunk_m = async_args.tiles_per_chunk_m;
        state_.chunk_signals     = async_args.chunk_signals;
        state_.tile_idx_pivot_m  = async_args.tile_idx_pivot_m;
    }

    /**
     * @brief Get the next work tile for this workgroup
     *
     * This function:
     * 1. Calculates the tile indices from linear work index
     * 2. Applies pivot to M dimension for hotspot spreading
     * 3. Waits for chunk signal if async mode is enabled
     * 4. Returns tile info
     */
    CK_TILE_DEVICE WorkTileInfo GetNextWorkTile()
    {
        const index_t linear_idx = state_.current_tile_linear;

        // Check if we've processed all tiles
        const index_t total_tiles = state_.total_tiles_m * state_.total_tiles_n;
        if(linear_idx >= total_tiles)
        {
            return WorkTileInfo::InvalidTile();
        }

        // Map linear index to 2D tile coordinates
        // Using row-major traversal (can be extended for different patterns)
        index_t tile_m = linear_idx / state_.total_tiles_n;
        index_t tile_n = linear_idx % state_.total_tiles_n;

        // Apply pivot to M dimension after basic mapping
        if(state_.tile_idx_pivot_m > 0)
        {
            tile_m = (tile_m + state_.tile_idx_pivot_m) % state_.total_tiles_m;
        }

        // Wait for async input readiness if enabled
        if(state_.chunk_signals != nullptr && state_.tiles_per_chunk_m > 0)
        {
            const index_t chunk_idx = tile_m / state_.tiles_per_chunk_m;

            // Producer lane waits for signal
            if(state_.is_mainloop_producer && threadIdx.x == 0)
            {
                wait_signal(state_.chunk_signals + chunk_idx);
            }

            // Synchronize all threads in workgroup after producer receives signal
            __builtin_amdgcn_s_barrier();
        }

        return WorkTileInfo{tile_m, tile_n, 0, true};
    }

    /**
     * @brief Advance to next work tile in persistent loop
     */
    CK_TILE_DEVICE void AdvanceToNextTile()
    {
        state_.current_tile_linear += state_.total_grid_size;
    }

    /**
     * @brief Check if this is the last tile for this workgroup
     */
    CK_TILE_DEVICE bool IsLastTile() const
    {
        const index_t total_tiles = state_.total_tiles_m * state_.total_tiles_n;
        return (state_.current_tile_linear + state_.total_grid_size) >= total_tiles;
    }

    /**
     * @brief Synchronization fence between persistent loop iterations
     *
     * Must be called before moving to next tile to ensure:
     * - All memory operations complete
     * - LDS can be safely reused
     */
    CK_TILE_DEVICE void IterationBoundaryFence() { iteration_boundary_fence(); }

    CK_TILE_DEVICE const SchedulerState& GetState() const { return state_; }

    private:
    SchedulerState state_;
};

} // namespace ck_tile
