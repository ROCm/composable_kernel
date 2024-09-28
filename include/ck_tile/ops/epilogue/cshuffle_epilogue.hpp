// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

#define CK_TILE_MAX_RANK 5

namespace ck_tile {

// this epilogue aiming to store a matrix with different layout from the shared memory to the global
// memory.
template <typename AccDataType_,
          typename ODataType_,
          bool kPadM_,
          bool kPadN_,
          index_t kRank_,
          index_t kPerm0,
          index_t kPerm1,
          index_t TileSize0,
          index_t TileSize1,
          index_t kPerm2    = 0,
          index_t kPerm3    = 0,
          index_t kPerm4    = 0,
          index_t TileSize2 = 0,
          index_t TileSize3 = 0,
          index_t TileSize4 = 0>
struct CShuffleEpilogueProblem
{
    using AccDataType                                = remove_cvref_t<AccDataType_>;
    using ODataType                                  = remove_cvref_t<ODataType_>;
    static constexpr bool kPadM                      = kPadM_;
    static constexpr bool kPadN                      = kPadN_;
    static constexpr index_t kRank                   = kRank_;
    static constexpr index_t kPerm[CK_TILE_MAX_RANK] = {kPerm0, kPerm1, kPerm2, kPerm3, kPerm4};
    static constexpr index_t tile_sizes[CK_TILE_MAX_RANK] = {
        TileSize0, TileSize1, TileSize2, TileSize3, TileSize4};
};

template <typename Problem_, typename Policy_ = void>
struct CShuffleEpilogue
{
    using Problem               = remove_cvref_t<Problem_>;
    using AccDataType           = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    const index_t kRank         = Problem::kRank;
    const index_t* kPerm        = Problem::kPerm;
    const index_t* tile_sizes   = Problem::tile_sizes;

    // No additional shared memory needed
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    template <typename ODramWindowTmp, typename OAccTile>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp, const OAccTile& o_acc_tile)
    {
        const auto& current_window_origin = o_dram_window_tmp.get_window_origin();

        // Compute the tile coordinates by dividing the window origin by the tile sizes
        index_t tile_coords[CK_TILE_MAX_RANK] = {0};
        for(index_t i = 0; i < kRank; ++i)
        {
            tile_coords[i] = current_window_origin[i] / tile_sizes[i];
            printf("The tile_coord is: %d", tile_coords[i]);
        }

        // Apply the permutation to the tile coordinates
        index_t permuted_tile_coords[CK_TILE_MAX_RANK];
        for(index_t i = 0; i < kRank; ++i)
        {
            permuted_tile_coords[i] = tile_coords[kPerm[i]];
            printf("The new permuted_tile_coords is: %d", permuted_tile_coords[i]);
        }

        // Compute the permuted window origin
        index_t permuted_window_origin[CK_TILE_MAX_RANK] = {0};
        for(index_t i = 0; i < kRank; ++i)
        {
            permuted_window_origin[i] = permuted_tile_coords[i] * tile_sizes[i];
            printf("The new permuted_window_origin is: %d", permuted_window_origin[i]);
        }

        typename ODramWindowTmp::BottomTensorIndex step = {};
        for(index_t i = 0; i < kRank; ++i)
        {
            step[i] = permuted_window_origin[i] - current_window_origin[i];
        }

        // Move the window
        move_tile_window(o_dram_window_tmp, step);

        // Store the tile data to the permuted location
        if constexpr(kPadM || kPadN)
        {
            store_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            buffer_store_fence();
        }
        else
        {
            store_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
        }
    }
};

} // namespace ck_tile
