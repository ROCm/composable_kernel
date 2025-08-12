// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file
 * GemmTilePartitioner allows customized mapping between a workgroup and the C-tile it computes.
 */

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/**
 * @brief Class providing 2D workgroup index mapping into 2D output GEMM C-tile space.
 *
 */
template <typename BlockGemmShapeType>
struct GemmTile2DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE GemmTile2DPartitioner() noexcept = delete;
    CK_TILE_HOST_DEVICE GemmTile2DPartitioner([[maybe_unused]] index_t M,
                                              [[maybe_unused]] index_t N) noexcept;

    /**
     * @brief Calculates GEMM kernel grid size.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     * @return dim3 Structure holding grid's X,Y and Z dimensions.
     */
    CK_TILE_HOST static auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        return dim3(GridDimX, GridDimY, 1);
    }

    /**
     * @brief Calculate number of loop iterations over GEMM's K dimension.
     *
     * @param K         GEMM's K dimension.
     * @return index_t  The number of loop iterations over K dimension.
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief The function returns 2D output tile space.
     * @param [in] blockIdx is blockIdx.x
     * @param [in] blockIdy is blockIdx.y
     * @return Returns the output tile indexes.
     */

    /**
     * @brief Calculate workgroup 2D index mapping into 2D output C-tile space.
     *
     * @param blockIdx      WGP's X index.
     * @param blockIdy      WGP's Y index.
     * @return const tuple<index_t, index_t>    Tuple containing 2D output C-tile index.
     */
    CK_TILE_DEVICE static auto GetOutputTileIndex(index_t blockIdx, index_t blockIdy) noexcept
        -> const tuple<index_t, index_t>
    {
        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdy);
        return make_tuple(iM, iN);
    }
};

/**
 * @brief Class providing 1D WGP index mapping into 2D output C-tile space.
 *
 * @tparam BlockGemmShape_  A class providing basic GEMM parameters. \link TileGemmShape
 */
template <typename BlockGemmShape_>
struct GemmTile1DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE GemmTile1DPartitioner() noexcept = delete;

    /**
     * @brief Construct a new GemmTile1DPartitioner object.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     */
    CK_TILE_HOST_DEVICE GemmTile1DPartitioner([[maybe_unused]] index_t M, index_t N) noexcept
    {
        N_ = N;
    }

    /**
     * @brief Calculates GEMM kernel grid size.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     * @return dim3 Structure holding grid's X,Y and Z dimensions.
     */
    CK_TILE_HOST_DEVICE static auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> index_t
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        return GridDimX * GridDimY;
    }

    /**
     * @brief Calculate number of loop iterations over GEMM's K dimension.
     *
     * @param K         GEMM's K dimension.
     * @return index_t  The number of loop iterations over K dimension.
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief Calculate workgroup 1D index mapping into 2D output C-tile space.
     *
     * @param blockIdx      WGP's index.
     * @return const tuple<index_t, index_t>    Tuple containing 2D output C-tile index.
     */
    CK_TILE_DEVICE static auto GetOutputTileIndex(index_t blockIdx) noexcept
        -> const tuple<index_t, index_t>
    {
        const index_t NBlocks = integer_divide_ceil(N_, NPerBlock);

        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx / NBlocks);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdx - iM * NBlocks);
        return make_tuple(iM, iN);
    }

    private:
    CK_TILE_DEVICE static index_t N_;
};

/**
 * @brief `GemmTile1DPartitioner::GetOutputTileIndex`'s std::false specialization,
 * checking expression validity in-place for ill-formed.
 */
template <typename, typename = void>
struct HasFnOneArgImpl : std::false_type
{
};

/**
 * @brief `GemmTile1DPartitioner::GetOutputTileIndex`'s std::true specialization,
 * checking expression validity in-place for well-formed.
 * @note: `1` - a constant value indicating the number of parameters in the function.
 */
template <typename T>
struct HasFnOneArgImpl<T, std::void_t<decltype(std::declval<T>().GetOutputTileIndex(1))>>
    : std::true_type
{
};

/**
 * @brief Struct used to calculate offseted tile indexes.
 * @note: The struct supports the 1D-Partitioner mechanism,
 * enable-if `GetOutputTileIndex`-fn is std::true_type when `GetOutputTileIndex`-fn is well-formed,
 * otherwise std::false_type.
 */
template <typename TilePartitioner,
          typename = typename std::enable_if_t<HasFnOneArgImpl<TilePartitioner>{}>>
struct OffsettedTile1DPartitioner
{
    /**
     * @brief The function subtracts the block's start (offset) from 1D raw-indexes.
     * @param [in] block_start Workgroup offset.
     * @param [in] M           Gemm's M dimension.
     * @param [in] N           Gemm's N dimension.
     * @return Returns a `tuple` [Im, In] with shifted index.
     */
    [[nodiscard]] CK_TILE_DEVICE static auto
    GetOffsetedTileIndex(index_t block_start, index_t M, index_t N) noexcept
        -> const tuple<index_t, index_t>
    {
        const auto [iM, iN] = TilePartitioner{M, N}.GetOutputTileIndex(blockIdx.x - block_start);
        return make_tuple(iM, iN);
    }

    /**
     * @brief The function subtracts the block's start (offset) from a given block index.
     * @param [in] block_start Workgroup offset.
     * @param [in] M           Gemm's M dimension.
     * @param [in] N           Gemm's N dimension.
     * @param [in] block_idx   Current block index of the workgroup.
     * @return Returns a `tuple` [Im, In] with shifted index.
     */
    [[nodiscard]] CK_TILE_DEVICE static auto
    GetOffsetedTileIndex(index_t block_start, index_t M, index_t N, index_t block_idx) noexcept
        -> const tuple<index_t, index_t>
    {
        const auto [iM, iN] = TilePartitioner{M, N}.GetOutputTileIndex(block_idx - block_start);
        return make_tuple(iM, iN);
    }
};

/**
 * @brief Class mapping 1D block index into 2D output tile space.
 *
 * @note It groups spatially workgroups in order to better utilize caches.
 *       It is using grouped Rows of column-vectors WGP pattern. It's optimized
 *       for gfx94x-like multiple-die chip.
 *
 * @tparam GroupNum - The number of big groups.
 * @tparam M01      - The number of groups in M dim within spatially local WGPs,
 *
 */
template <typename BlockGemmShapeType, index_t GroupNum, index_t M01>
struct GemmSpatiallyLocalTilePartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE GemmSpatiallyLocalTilePartitioner() noexcept = delete;
    CK_TILE_HOST_DEVICE GemmSpatiallyLocalTilePartitioner(index_t M_, index_t N_) noexcept
        : M(M_), N(N_)
    {
    }

    /**
     * @brief Calculates GEMM kernel grid size.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     * @return index_t A total number of workgroups.
     */
    CK_TILE_HOST_DEVICE static auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> index_t
    {
        const index_t GridDimX = integer_divide_ceil(M, MPerBlock);
        const index_t GridDimY = integer_divide_ceil(N, NPerBlock);
        return GridDimX * GridDimY;
    }

    /**
     * @brief Calculate number of loop iterations over GEMM's K dimension.
     *
     * @param K         GEMM's K dimension.
     * @return index_t  The number of loop iterations over K dimension.
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief Calculate workgroup 1D index mapping into 2D output C-tile space.
     *
     * @param [in] block_1d_id      WGP's index.
     * @return const tuple<index_t, index_t>    Tuple containing 2D output C-tile index.
     */
    CK_TILE_DEVICE auto GetOutputTileIndex(index_t block_1d_id) noexcept
        -> const tuple<index_t, index_t>
    {
        const auto M0 = integer_divide_ceil(M, MPerBlock);
        const auto N0 = integer_divide_ceil(N, NPerBlock);

        if(M0 == 1)
        {
            return make_tuple(0, block_1d_id);
        }
        else if(N0 == 1)
        {
            return make_tuple(block_1d_id, 0);
        }
        // block_1d_id = block_1d_id % (M0 * N0); // swallow batch index
        else
        {
            const auto group_size    = integer_divide_ceil(M0 * N0, GroupNum);
            const auto big_group_num = GroupNum - (group_size * GroupNum - M0 * N0);
            const auto group_id_y    = block_1d_id / GroupNum;
            const auto group_id_x    = block_1d_id - group_id_y * GroupNum;
            const auto remap_block_1d_id =
                group_id_x <= big_group_num
                    ? group_id_x * group_size + group_id_y
                    : group_id_x * group_size + big_group_num - group_id_x + group_id_y;

            const index_t idx_M0 = remap_block_1d_id / N0;
            const index_t idx_N0 = remap_block_1d_id - idx_M0 * N0;

            const index_t M0_tmp     = M0 / M01;
            const index_t M0_mod_M01 = M0 - M0_tmp * M01;

            const auto M01_adapt = (idx_M0 < M0 - M0_mod_M01) ? M01 : M0_mod_M01;

            const index_t idx_M00          = idx_M0 / M01;
            const index_t idx_M01          = idx_M0 - idx_M00 * M01;
            const index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

            /**
             *                        idxN0
             *
             *           |<               mtx   N                 >|
             *
             *             NPerBlock   NPerBlock   NPerBlock   NPerBlock
             *                N_0         N_1        N_2         N_3
             *       -   |-----------|-----------|-----------|-----|-----|-
             *       ^   | -   -  0  |/---->  2  |           |     |     |
             *           | |   |     /     |     |           |     |     |  M_0  MPerBlock
             *           | M   |    /|     |     |           |     |     |
             *           |-0---|---/-|-----|-----|-----------|-----|-----|-
             *           | 1   |  /  |     |     |  blockid  |     |     |
             * idxM0     | |   | /   |     V     |     5     |     |     |  M_1  MPerBlock
             *           | -   V   1 |     -  3  |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *    mtx M  |           |           |           |     |     |
             *           |           |           |           |     |     |  M_2  MPerBlock
             *           |           |           |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *           |           |           |           |     |     |
             *           |           |           |           |     |     |  M_3  MPerBlock
             *           |           |           |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *       V   |           |           |           |     |     |
             *       -   |-----------|-----------|-----------|-----|-----|- M_4  MPerBlock
             *           |           |           |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *  Example:
             *   assume:
             *      M0 = 5
             *      N0 = 4
             *      block_1d_id = 5
             *      M01 = 2
             *
             *   idx_N0 = 1
             *   idx_M0 = 1
             *   M01_adapt = 2
             *   idx_M00 = 0
             *   idx_M01 = 1
             *   idx_N0_M01_local = 5
             *   output {1, 2}
             */

            const index_t N_out           = idx_N0_M01_local / M01_adapt;
            const index_t idx_loc_mod_M01 = idx_N0_M01_local - N_out * M01_adapt;

            return make_tuple(idx_loc_mod_M01 + idx_M00 * M01, N_out);
        }
    }

    private:
    index_t M;
    index_t N;
};

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

        total_tiles_      = num_tile_m_ * num_tile_n_;
        total_work_units_ = total_tiles_ * num_tile_k_;
    }

    /**
     * @brief Calculate optimal grid size for Stream-K (always assumes Stream-K usage)
     */
    CK_TILE_HOST static auto GridSize(index_t M, index_t N, index_t K) noexcept -> index_t
    {
        const auto [target_blocks, work_per_block, big_blocks, total_work] =
            StreamKWorkAnalysis::CalculateWorkDistribution(
                M, N, K, MPerBlock, NPerBlock, KPerBlock);
        return target_blocks;
    }

    /**
     * @brief Calculate number of loop iterations over K dimension for given work unit
     */
    CK_TILE_HOST_DEVICE auto GetLoopNum(index_t work_start, index_t work_end) const noexcept
        -> index_t
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

        if(block_id < remainder_work)
        {
            // Blocks with extra work
            work_start = block_id * (work_per_block + 1);
            work_end   = work_start + work_per_block + 1;
        }
        else
        {
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
        const index_t k_tile            = work_idx / tiles_per_k_slice;
        const index_t tile_idx          = work_idx % tiles_per_k_slice;

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
    CK_TILE_HOST_DEVICE static bool ShouldUseStreamK(index_t M,
                                                     index_t N,
                                                     index_t K,
                                                     index_t MPerBlock,
                                                     index_t NPerBlock,
                                                     index_t KPerBlock) noexcept
    {
        const index_t num_tile_k = (K + KPerBlock - 1) / KPerBlock;
        const index_t num_tile_mn =
            ((M + MPerBlock - 1) / MPerBlock) * ((N + NPerBlock - 1) / NPerBlock);

        // Use Stream-K when:
        // 1. K dimension has multiple tiles (more than 2)
        // 2. Total work significantly exceeds output tiles
        // 3. Problem size is large enough to benefit
        return (num_tile_k > 2) && (num_tile_k * num_tile_mn > num_tile_mn * 2) &&
               (num_tile_mn * num_tile_k > 64);
    }

    CK_TILE_HOST_DEVICE static auto CalculateWorkDistribution(index_t M,
                                                              index_t N,
                                                              index_t K,
                                                              index_t MPerBlock,
                                                              index_t NPerBlock,
                                                              index_t KPerBlock,
                                                              index_t target_blocks = 0) noexcept
    {
        const index_t num_tile_m = (M + MPerBlock - 1) / MPerBlock;
        const index_t num_tile_n = (N + NPerBlock - 1) / NPerBlock;
        const index_t num_tile_k = (K + KPerBlock - 1) / KPerBlock;

        const index_t total_output_tiles = num_tile_m * num_tile_n;
        const index_t total_work_units   = total_output_tiles * num_tile_k;

        // If target_blocks is 0, calculate optimal blocks
        if(target_blocks == 0)
        {
            // Heuristic: aim for 2-4x more blocks than output tiles, but cap at total work
            target_blocks = min(total_work_units, total_output_tiles * 4);
            target_blocks =
                max(target_blocks, total_output_tiles); // At least as many as output tiles
        }

        const index_t work_per_block = total_work_units / target_blocks;
        const index_t big_blocks     = total_work_units % target_blocks;

        return make_tuple(target_blocks, work_per_block, big_blocks, total_work_units);
    }
};

} // namespace ck_tile
