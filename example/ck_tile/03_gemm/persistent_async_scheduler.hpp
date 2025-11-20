// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
