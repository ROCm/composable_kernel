// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/workgroup_barrier.hpp"

namespace ck {

enum struct WorkSchedulingPolicy
{
    StridedTileLoop
};

///
/// @brief      This class describes a strided reduction tile loop work scheduling.
///
///
/// @par Overview
///     This work scheduling policy assume linear mapping (with stride) of workgroups along
///     the reduced dimension. In GEMM problem this mean that consecutive workgroups are mapped
///     to strided data tiles along K dimension. This can be obtained using i.e.
///     @see BlockToCTileMap_ReduceKSplit.
///
/// @par Synchronization
///     All workgroups aligned along particular reduced dimension have to reduce their partial
///     results. In order to do that there's a need to use global flags and atomics to communicate
///     between those workgroups.
///
class StridedReductionTileLoop
{
    public:
    __device__ StridedReductionTileLoop(index_t tile_count, uint32_t* const __restrict__ p_flags)
        : tile_count_{tile_count},
          tiles_per_block_{(tile_count_ + get_grid_size() - 1) / get_grid_size()},
          tile_id_{get_block_1d_id() * tiles_per_block_},
          block_tile_idx_{0},
          finished_block_flags_{p_flags}
    {
    }

    __device__ bool HasTile() const
    {
        return tile_id_ < tile_count_ && block_tile_idx_ < tiles_per_block_;
    }

    __device__ bool GetNextTile()
    {
        tile_id_++;
        block_tile_idx_++;
        return HasTile();
    }

    /// @brief Returns the number of next k-tiles to process.
    /// @param[in]  k_tiles     The number of tiles in the reduced dimension.
    /// @param[in]  k_tile_idx  Current k-tile index.
    /// @return The number of next k-tiles to process.
    __device__ index_t GetNextKTiles(index_t k_tiles, index_t k_tile_idx)
    {
        index_t k_tiles_left     = k_tiles - k_tile_idx;
        index_t block_tiles_left = tiles_per_block_ - block_tile_idx_;
        index_t next_k_tiles = k_tiles_left <= block_tiles_left ? k_tiles_left : block_tiles_left;

        tile_id_ += next_k_tiles;
        block_tile_idx_ += next_k_tiles;
        return next_k_tiles;
    }

    __device__ index_t GetFlagCount() const { return get_grid_size(); }

    ///
    /// @brief      Get this workgroup flag index.
    ///
    /// @return     The workgroup flag index.
    ///
    __device__ uint32_t GetWorkgroupFlagIdx() const { return static_cast<uint32_t>(blockIdx.x); }

    ///
    /// @brief      Flag each workgroup that has finished its work.
    ///
    __device__ void FlagFinished() { finished_block_flags_.inc(GetWorkgroupFlagIdx()); }

    ///
    /// @brief      Wait until each workgroup has finished its work.
    ///
    /// @param[in]  k_tiles     The number of tiles in the reduced dimension.
    /// @param[in]  k_tile_idx  The currently processed tile k index.
    ///
    /// @return     The number of neighbours.
    ///
    __device__ index_t WaitForNeighbours(index_t k_tiles, index_t k_tile_idx)
    {
        // We have to wait for all workgroups to finish their partial results.
        // First count how many "neighbour" workgroups we have to check.
        index_t neighbour_count = 0;
        if(tiles_per_block_ < k_tiles)
        {
            // Since we can have deviation (+/-1) in neighbours number
            // we calculate how many workgroups are needed to process the k-tiles left.
            neighbour_count = (k_tiles - k_tile_idx - 1 + tiles_per_block_ - 1) / tiles_per_block_;
        }
        // If we have more tiles to process than the reduction dimension size,
        // then the number of neighbours depends on first K-tile workgroup block tile idx.
        else
        {
            if(block_tile_idx_ == tiles_per_block_)
            {
                // If we just finished work per workgroup then check at which k-idx we are.
                neighbour_count = (k_tile_idx < k_tiles - 1) ? 1 : 0;
            }
            else
            {
                // If we have still tiles to process then it means that we already processed
                // whole K-dim.
                neighbour_count = 0;
            }
        }

        if(neighbour_count > 0)
        {
            // Check if this workgroup's flag is also set (i = 0)
            for(index_t i = 0; i <= neighbour_count; ++i)
            {
                finished_block_flags_.wait_eq(GetWorkgroupFlagIdx() + i, 1);
            }
        }

        return neighbour_count;
    }

    ///
    /// @brief      Wait until reduction workgroup has finished its work.
    ///
    __device__ void WaitForReduction()
    {
        // Wait untill my counter has been reset.
        finished_block_flags_.wait_eq(GetWorkgroupFlagIdx(), 0);
    }

    ///
    /// @brief      Reset flag counter to zero.
    ///
    /// @param[in]  neighbour_count     The number of peer workgroups.
    ///
    __device__ void Reset(index_t neighbour_count)
    {
        for(index_t i = 0; i <= neighbour_count; ++i)
        {
            finished_block_flags_.reset(GetWorkgroupFlagIdx() + i);
        }
    }

    ///
    /// @brief      Gets the flag value.
    ///
    __device__ uint32_t GetFlagValue() const
    {
        return finished_block_flags_.ld(GetWorkgroupFlagIdx());
    }

    const index_t tile_count_;
    const index_t tiles_per_block_;
    index_t tile_id_;
    index_t block_tile_idx_;
    workgroup_barrier finished_block_flags_;
};

} // namespace ck
