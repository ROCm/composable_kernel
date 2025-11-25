// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/**
 * @brief Safe iteration boundary fence for persistent kernels
 *
 * This function ensures memory consistency between iterations in persistent loops by:
 * - Waiting for all vector memory operations to complete (vmcnt=0)
 * - Waiting for all LDS/GDS operations to complete (lgkmcnt=0)
 * - Synchronizing all workgroup threads via barrier
 *
 * This prevents race conditions when reusing LDS or moving to the next tile.
 */
CK_TILE_DEVICE static void iteration_boundary_fence()
{
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
}

/**
 * @brief Wait for chunk readiness signal (producer-consumer synchronization)
 *
 * This function implements producer-consumer synchronization for async input readiness:
 * - One lane polls the chunk_signals[chunk_idx] flag with acquire semantics
 * - When signal becomes ready (value == 1), a workgroup barrier releases all threads
 *
 * @param chunk_signals Device pointer to global chunk readiness flags array
 * @param chunk_idx Index of the chunk to wait for
 *
 * @note Only lane 0 performs the polling to minimize global memory traffic
 * @note Uses acquire semantics to ensure proper memory ordering
 */
CK_TILE_DEVICE static void wait_chunk_signal(const uint32_t* chunk_signals, index_t chunk_idx)
{
    // Only lane 0 polls the signal to minimize global memory traffic
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        volatile const uint32_t* signal_ptr = chunk_signals + chunk_idx;
        
        // Poll until chunk is ready (signal == 1)
        // Use acquire semantics for proper memory ordering
        uint32_t signal_value;
        do {
            signal_value = __builtin_nontemporal_load(signal_ptr);
            __builtin_amdgcn_s_sleep(1); // Brief sleep to reduce contention
        } while(signal_value == 0);
        
        // Memory fence with acquire semantics
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    
    // Barrier to release all threads in the workgroup
    __builtin_amdgcn_s_barrier();
}

} // namespace ck_tile
