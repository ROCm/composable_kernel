// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename FusedMoeTileShape_>
struct FusedMoeTilePartitioner_PersistentSplitD
{
    using FusedMoeTileShape = ck_tile::remove_cvref_t<FusedMoeTileShape_>;

    static constexpr index_t kM_a = FusedMoeTileShape::kM_a;
    static constexpr index_t kN_g = FusedMoeTileShape::kN_g;
    static constexpr index_t kN_u = FusedMoeTileShape::kN_u;
    static constexpr index_t kK_a = FusedMoeTileShape::kK_a;
    static constexpr index_t kN_d = FusedMoeTileShape::kN_d;

    static constexpr const char* name = "psd"; // expert x hidden

    CK_TILE_DEVICE auto operator()(ck_tile::index_t tile_id,
                                   ck_tile::index_t /*num_sorted_tiles*/,
                                   ck_tile::index_t hidden_size)
    {
        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        const index_t num_hidden_tiles = ck_tile::integer_divide_ceil(hidden_size, kN_g);
        const auto [sorted_tile_id, hidden_tile_id] = f(tile_id, num_hidden_tiles);

        return ck_tile::make_tuple(sorted_tile_id, hidden_tile_id);
    }

    // persistent
    CK_TILE_HOST static constexpr auto GridSize(index_t num_cu, index_t blocks_per_cu)
    {
        // TODO: this may need tuning
        index_t grids = num_cu * blocks_per_cu;
        return dim3(grids);
    }
};

template <typename FusedMoeTileShape_>
struct FusedMoeTilePartitioner_Linear
{
    using Shape = ck_tile::remove_cvref_t<FusedMoeTileShape_>;

    static constexpr const char* name = "2d"; // expert x hidden

    CK_TILE_DEVICE auto operator()(ck_tile::index_t /*num_sorted_tiles*/,
                                   ck_tile::index_t /*hidden_size*/))
    {
        index_t i_n = blockIdx.x;
        index_t i_m = blockIdx.y;

        return ck_tile::make_tuple(i_m, i_n);
    }

    // persistent
    CK_TILE_HOST static constexpr auto GridSize(index_t max_tokens, index_t hidden_size)
    {
        // TODO: this may need tuning
        index_t grids = num_cu * blocks_per_cu;
        index_t ms    = ck_tile::integer_divide_ceil(max_tokens, Shape::kBlockM_0);
        index_t ns    = ck_tile::integer_divide_ceil(hidden_size, Shape::kBlockN_0);
        return dim3(ns, ms, 1);
    }
};

} // namespace ck_tile
