// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file tile_load_store_microkernels.hpp
 * @brief Generic tile store/load microkernels.
 *
 * Setup::create() must return:
 *   - For StoreTile: tuple<window, tile>
 *   - For LoadTile: window
 */

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename Setup>
struct StoreTile
{
    static constexpr index_t kBlockSize = Setup::kBlockSize;

    CK_TILE_DEVICE void operator()() const
    {
        auto [window, tile] = Setup::create();
        store_tile(window, tile);
        block_sync_lds();
    }
};

template <typename Setup>
struct LoadTile
{
    static constexpr index_t kBlockSize = Setup::kBlockSize;

    CK_TILE_DEVICE void operator()() const
    {
        auto window                         = Setup::create();
        [[maybe_unused]] volatile auto tile = load_tile(window);
        block_sync_lds();
    }
};

} // namespace ck_tile
