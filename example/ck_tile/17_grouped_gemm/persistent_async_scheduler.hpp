// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

/**
 * @brief Arguments for Persistent Async GEMM scheduling
 *
 * This structure contains parameters for producer-consumer synchronization
 * in persistent GEMM kernels with asynchronous input readiness.
 */
struct PersistentAsyncArgs
{
    /// Number of M tiles per chunk (granularity of async readiness signaling)
    index_t tiles_per_chunk_m = 0;

    /// Device pointer to global chunk readiness flags (1 = ready, 0 = not ready)
    uint32_t* chunk_signals = nullptr;

    /// Pivot offset for M dimension (for hotspot spreading in tile scheduling)
    index_t tile_idx_pivot_m = 0;

    /// Enable/disable async input signaling (false = disabled, true = enabled)
    bool enable_async = false;

    CK_TILE_HOST_DEVICE PersistentAsyncArgs() = default;

    CK_TILE_HOST_DEVICE PersistentAsyncArgs(index_t tiles_per_chunk_m_,
                                            uint32_t* chunk_signals_,
                                            index_t tile_idx_pivot_m_,
                                            bool enable_async_ = false)
        : tiles_per_chunk_m(tiles_per_chunk_m_),
          chunk_signals(chunk_signals_),
          tile_idx_pivot_m(tile_idx_pivot_m_),
          enable_async(enable_async_)
    {
    }
};
} // namespace ck_tile
