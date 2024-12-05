// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

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
          bool kTilePermute_,
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
    static constexpr bool kTilePermute               = kTilePermute_;
    static constexpr index_t kRank                   = kRank_;
    static constexpr index_t kPerm[CK_TILE_MAX_RANK] = {kPerm0, kPerm1, kPerm2, kPerm3, kPerm4};
    static constexpr index_t tile_sizes[CK_TILE_MAX_RANK] = {
        TileSize0, TileSize1, TileSize2, TileSize3, TileSize4};
};

template <typename Problem_, typename Policy_ = void>
struct CShuffleEpilogue
{
    using Problem                      = remove_cvref_t<Problem_>;
    using AccDataType                  = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                    = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadM        = Problem::kPadM;
    static constexpr bool kPadN        = Problem::kPadN;
    const index_t* kPerm               = Problem::kPerm;
    static constexpr bool kTilePermute = Problem::kTilePermute;
    static constexpr index_t kRank     = Problem::kRank;
    const index_t* tile_sizes          = Problem::tile_sizes;

    // No additional shared memory needed
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    CK_TILE_HOST_DEVICE static constexpr bool IsOutputTransposed()
    {
        // TODO: At now CShuffle doesn't allow to vector store after permute.
        //       It should be fixed and this function should return true.
        return false;
    }

    template <typename OAccTile>
    CK_TILE_DEVICE void permute_tile_data(OAccTile& o_acc_tile)
    {
        using DataType = typename OAccTile::DataType;

        // Get thread buffer
        auto& thread_buf = o_acc_tile.get_thread_buffer();

        // Create a temporary buffer to hold the permuted data
        thread_buffer<DataType, OAccTile::kThreadElementSpaceSize> permuted_thread_buf;

        // Get the lengths of each dimension
        auto thread_tensor_lengths = o_acc_tile.get_lengths();

        // Total number of elements
        index_t total_elements = OAccTile::kThreadElementSpaceSize;

        // Iterate over all elements
        for(index_t linear_idx = 0; linear_idx < total_elements; ++linear_idx)
        {
            // Convert linear index to multi-dimensional indices
            array<index_t, kRank> indices;
            index_t remaining = linear_idx;
            static_for<0, kRank, 1>{}([&](auto i) {
                constexpr auto rev_i = kRank - 1 - i;
                indices(rev_i)       = remaining % thread_tensor_lengths.get(number<rev_i>{});
                remaining /= thread_tensor_lengths.get(number<rev_i>{});
            });

            // Apply the permutation
            array<index_t, kRank> permuted_indices;
            static_for<0, kRank, 1>{}(
                [&](auto i) { permuted_indices(i) = indices.get(number<Problem::kPerm[i]>{}); });

            // Compute offsets
            index_t dst_offset = 0;
            index_t stride     = 1;

            static_for<0, kRank, 1>{}([&](auto i) {
                constexpr auto rev_i = kRank - 1 - i;
                dst_offset += permuted_indices[rev_i] * stride;
                stride *= thread_tensor_lengths.get(number<rev_i>{});
            });

            // Move the data
            permuted_thread_buf(dst_offset) = thread_buf[linear_idx];
        }

        // Copy the permuted data back to the original thread buffer
        for(index_t i = 0; i < total_elements; ++i)
        {
            thread_buf.set_as(i, permuted_thread_buf.get(i));
        }
    }

    template <typename ODramWindowTmp,
              typename OAccTile,
              memory_operation_enum out_memory_data_op = memory_operation_enum::set>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp, OAccTile& o_acc_tile)
    {
        const auto& current_window_origin = o_dram_window_tmp.get_window_origin();

        // Compute the tile coordinates by dividing the window origin by the tile sizes
        index_t tile_coords[CK_TILE_MAX_RANK] = {0};
        for(index_t i = 0; i < kRank; ++i)
        {
            tile_coords[i] = current_window_origin[i] / tile_sizes[i];
            // printf("The tile_coord is: %d", tile_coords[i]);
        }

        // Apply the permutation to the tile coordinates
        index_t permuted_tile_coords[CK_TILE_MAX_RANK];
        for(index_t i = 0; i < kRank; ++i)
        {
            permuted_tile_coords[i] = tile_coords[kPerm[i]];
            // printf("The new permuted_tile_coords is: %d", permuted_tile_coords[i]);
        }

        // Compute the permuted window origin
        index_t permuted_window_origin[CK_TILE_MAX_RANK] = {0};
        for(index_t i = 0; i < kRank; ++i)
        {
            permuted_window_origin[i] = permuted_tile_coords[i] * tile_sizes[i];
            // printf("The new permuted_window_origin is: %d", permuted_window_origin[i]);
        }

        typename ODramWindowTmp::BottomTensorIndex step = {};
        for(index_t i = 0; i < kRank; ++i)
        {
            step[i] = permuted_window_origin[i] - current_window_origin[i];
        }

        // Move the window
        move_tile_window(o_dram_window_tmp, step);

        // Permute the data within the tile if necessary
        if constexpr(kTilePermute)
        {
            permute_tile_data(o_acc_tile);
        }

        // Store the tile data to the permuted location
        if constexpr(kPadM || kPadN)
        {
            if constexpr(out_memory_data_op == memory_operation_enum::set)
            {
                store_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            else
            {
                update_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            buffer_store_fence();
        }
        else
        {
            if constexpr(out_memory_data_op == memory_operation_enum::set)
            {
                store_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            else
            {
                update_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
        }
    }
};

} // namespace ck_tile
