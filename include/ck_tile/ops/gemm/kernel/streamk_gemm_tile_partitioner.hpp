/ SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"

namespace ck_tile {

/**
 * @brief Stream-K tile partitioner that dynamically balances work across workgroups
 * 
 * This partitioner implements Stream-K algorithm which decomposes the GEMM problem
 * into smaller work units and distributes them more evenly across available blocks,
 * improving load balancing especially for cases where the K dimension is large.
 */
template <typename BlockGemmShape_>
struct StreamKTilePartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE StreamKTilePartitioner() noexcept = delete;

    /**
     * @brief Construct Stream-K tile partitioner with problem dimensions
     */
    CK_TILE_HOST_DEVICE StreamKTilePartitioner(index_t M, index_t N, index_t K) noexcept
        : M_(M), N_(N), K_(K)
    {
        num_tile_m_ = (M + MPerBlock - 1) / MPerBlock;
        num_tile_n_ = (N + NPerBlock - 1) / NPerBlock;
        num_tile_k_ = (K + KPerBlock - 1) / KPerBlock;
        
        total_tiles_ = num_tile_m_ * num_tile_n_;
        total_work_units_ = total_tiles_ * num_tile_k_;
    }

    /**
     * @brief Calculate optimal grid size for Stream-K (always assumes Stream-K usage)
     */
    CK_TILE_HOST static auto GridSize(index_t M, index_t N, index_t K) noexcept -> index_t
    {
        const auto [target_blocks, work_per_block, big_blocks, total_work] = 
            StreamKWorkAnalysis::CalculateWorkDistribution(M, N, K, MPerBlock, NPerBlock, KPerBlock);
        return target_blocks;
    }

    /**
     * @brief Calculate number of loop iterations over K dimension for given work unit
     */
    CK_TILE_HOST_DEVICE auto GetLoopNum(index_t work_start, index_t work_end) const noexcept -> index_t
    {
        const index_t work_units = work_end - work_start;
        // Each work unit represents one K-tile of computation
        return 1; // Stream-K processes one K-slice at a time
    }

    /**
     * @brief Get work range for a given block ID
     */
    CK_TILE_DEVICE auto GetWorkRange(index_t block_id, index_t total_blocks) const noexcept
        -> const tuple<index_t, index_t>
    {
        const index_t work_per_block = total_work_units_ / total_blocks;
        const index_t remainder_work = total_work_units_ % total_blocks;
        
        index_t work_start, work_end;
        
        if (block_id < remainder_work) {
            // Blocks with extra work
            work_start = block_id * (work_per_block + 1);
            work_end = work_start + work_per_block + 1;
        } else {
            // Regular blocks
            work_start = remainder_work * (work_per_block + 1) + 
                        (block_id - remainder_work) * work_per_block;
            work_end = work_start + work_per_block;
        }
        
        return make_tuple(work_start, work_end);
    }

    /**
     * @brief Convert linear work index to 3D tile coordinates (M, N, K)
     */
    CK_TILE_DEVICE auto WorkIndexToTileCoords(index_t work_idx) const noexcept
        -> const tuple<index_t, index_t, index_t>
    {
        const index_t tiles_per_k_slice = total_tiles_;
        const index_t k_tile = work_idx / tiles_per_k_slice;
        const index_t tile_idx = work_idx % tiles_per_k_slice;
        
        const index_t m_tile = tile_idx / num_tile_n_;
        const index_t n_tile = tile_idx % num_tile_n_;
        
        return make_tuple(m_tile, n_tile, k_tile);
    }

    /**
     * @brief Get output tile index for standard 2D mapping (compatibility)
     */
    CK_TILE_DEVICE auto GetOutputTileIndex(index_t tile_m, index_t tile_n) const noexcept
        -> const tuple<index_t, index_t>
    {
        return make_tuple(tile_m, tile_n);
    }

    /**
     * @brief Check if this partitioner should be used (always true for Stream-K partitioner)
     */
    CK_TILE_HOST_DEVICE bool ShouldUseStreamK() const noexcept
    {
        // This partitioner is specifically for Stream-K, so always return true
        // The decision to use Stream-K should be made before instantiating this partitioner
        return true;
    }

    /**
     * @brief Get the number of blocks that will have extra work
     */
    CK_TILE_HOST_DEVICE index_t GetBigBlockCount(index_t total_blocks) const noexcept
    {
        return total_work_units_ % total_blocks;
    }

    /**
     * @brief Get work units per block (for regular blocks)
     */
    CK_TILE_HOST_DEVICE index_t GetWorkPerBlock(index_t total_blocks) const noexcept
    {
        return total_work_units_ / total_blocks;
    }

    // Getters for problem dimensions
    CK_TILE_HOST_DEVICE index_t GetNumTileM() const noexcept { return num_tile_m_; }
    CK_TILE_HOST_DEVICE index_t GetNumTileN() const noexcept { return num_tile_n_; }
    CK_TILE_HOST_DEVICE index_t GetNumTileK() const noexcept { return num_tile_k_; }
    CK_TILE_HOST_DEVICE index_t GetTotalTiles() const noexcept { return total_tiles_; }
    CK_TILE_HOST_DEVICE index_t GetTotalWorkUnits() const noexcept { return total_work_units_; }

private:
    index_t M_, N_, K_;
    index_t num_tile_m_, num_tile_n_, num_tile_k_;
    index_t total_tiles_;
    index_t total_work_units_;
};

/**
 * @brief Static helper functions for Stream-K work analysis
 */
struct StreamKWorkAnalysis
{
    CK_TILE_HOST_DEVICE static bool ShouldUseStreamK(index_t M, index_t N, index_t K,
                                                     index_t MPerBlock, index_t NPerBlock, index_t KPerBlock) noexcept
    {
        const index_t num_tile_k = (K + KPerBlock - 1) / KPerBlock;
        const index_t num_tile_mn = ((M + MPerBlock - 1) / MPerBlock) * 
                                   ((N + NPerBlock - 1) / NPerBlock);
        
        // Use Stream-K when:
        // 1. K dimension has multiple tiles (more than 2)
        // 2. Total work significantly exceeds output tiles
        // 3. Problem size is large enough to benefit
        return (num_tile_k > 2) && 
               (num_tile_k * num_tile_mn > num_tile_mn * 2) &&
               (num_tile_mn * num_tile_k > 64);
    }

    CK_TILE_HOST_DEVICE static auto CalculateWorkDistribution(index_t M, index_t N, index_t K,
                                                              index_t MPerBlock, index_t NPerBlock, index_t KPerBlock,
                                                              index_t target_blocks = 0) noexcept
    {
        const index_t num_tile_m = (M + MPerBlock - 1) / MPerBlock;
        const index_t num_tile_n = (N + NPerBlock - 1) / NPerBlock;
        const index_t num_tile_k = (K + KPerBlock - 1) / KPerBlock;
        
        const index_t total_output_tiles = num_tile_m * num_tile_n;
        const index_t total_work_units = total_output_tiles * num_tile_k;
        
        // If target_blocks is 0, calculate optimal blocks
        if (target_blocks == 0) {
            // Heuristic: aim for 2-4x more blocks than output tiles, but cap at total work
            target_blocks = min(total_work_units, total_output_tiles * 4);
            target_blocks = max(target_blocks, total_output_tiles); // At least as many as output tiles
        }
        
        const index_t work_per_block = total_work_units / target_blocks;
        const index_t big_blocks = total_work_units % target_blocks;
        
        return make_tuple(target_blocks, work_per_block, big_blocks, total_work_units);
    }
};

} // namespace ck_tile