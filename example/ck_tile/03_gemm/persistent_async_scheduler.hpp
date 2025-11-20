// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

__device__ wait_signal(uint32_t* signal_addr)
{
    uint32_t ready = __atomic_load_n(signal_addr, __ATOMIC_ACQUIRE);
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

        ready = __atomic_load_n(signal_addr, __ATOMIC_ACQUIRE);
    }
}

struct PersistentAsyncArgs
{
    index_t tiles_per_chunk_m = 0;

    uint32_t* chunk_signals = nullptr;

    index_t tile_idx_pivot_m = 0;

    PersistentAsyncArgs(index_t tiles_per_chunk_m_,
                        uint32_t* chunk_signals_,
                        index_t tile_idx_pivot_m_,
                        bool enable_async_)
        : tiles_per_chunk_m(tiles_per_chunk_m_),
          chunk_signals(chunk_signals_),
          tile_idx_pivot_m(tile_idx_pivot_m_)
    {
    }
};
} // namespace ck_tile