// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * GemmTilePartitioner allows customized mapping between a workgroup and the C-tile it computes.
 */

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

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
    CK_TILE_DEVICE static auto
    GetOutputTileIndex(index_t blockIdx, index_t blockIdy) noexcept -> const tuple<index_t, index_t>
    {
        const index_t iM = amd_wave_read_first_lane(blockIdx);
        const index_t iN = amd_wave_read_first_lane(blockIdy);
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
    CK_TILE_DEVICE static auto
    GetOutputTileIndex(index_t blockIdx) noexcept -> const tuple<index_t, index_t>
    {
        const index_t NBlocks = integer_divide_ceil(N_, NPerBlock);

        const index_t iM = amd_wave_read_first_lane(blockIdx / NBlocks);
        const index_t iN = amd_wave_read_first_lane(blockIdx - iM * NBlocks);
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
    [[nodiscard]] CK_TILE_DEVICE static auto GetOffsetedTileIndex(
        index_t block_start, index_t M, index_t N) noexcept -> const tuple<index_t, index_t>
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
     * @brief XCDs access ids in round robin format, this function remaps the 1D ids to continguous
     * XCD segments
     *
     * @param block_1d_id       grid 1D id
     * @param total_num_tiles   size of the 1D grid
     * @param NUM_XCDS          number of XCDs
     * @return index_t  The id after XCD remap
     */
    CK_TILE_HOST_DEVICE static auto
    RemapXCD(index_t block_1d_id, index_t total_num_tiles, index_t NUM_XCDS = 8) noexcept -> index_t
    {
        // Number of ids per XCD in the new arrangement
        index_t ids_per_xcd = (total_num_tiles + NUM_XCDS - 1) / NUM_XCDS;

        // When total_num_tiles cannot divide NUM_XCDS, some xcds will have
        // ids_per_xcd ids, the other will have ids_per_xcd - 1 ids.
        // We calculate the number of xcds that have ids_per_xcd ids as tall_xcds
        index_t tall_xcds = total_num_tiles % NUM_XCDS;
        tall_xcds         = (tall_xcds == 0) ? NUM_XCDS : tall_xcds;

        // Compute current XCD and local id within the XCD
        index_t xcd      = block_1d_id % NUM_XCDS;
        index_t local_id = block_1d_id / NUM_XCDS;

        // Calculate new id based on the new grouping
        if(xcd < tall_xcds)
        {
            block_1d_id = xcd * ids_per_xcd + local_id;
        }
        else
        {
            block_1d_id =
                tall_xcds * ids_per_xcd + (xcd - tall_xcds) * (ids_per_xcd - 1) + local_id;
        }

        /**
         * original ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
         * XCD 0 gets: [0, 8], XCD 1 gets: [1, 9], ...
         *
         * post-remap ids: [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
         * XCD 0 gets: [0, 1], XCD 1 gets: [2, 3], ...
         *
         * after remap the ids are continguous on each XCD
         */
        return block_1d_id;
    }

    /**
     * @brief Calculate workgroup 1D index mapping into 2D output C-tile space.
     *
     * @param [in] block_1d_id      WGP's index.
     * @return const tuple<index_t, index_t>    Tuple containing 2D output C-tile index.
     */
    CK_TILE_DEVICE auto
    GetOutputTileIndex(index_t block_1d_id) noexcept -> const tuple<index_t, index_t>
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

enum class ClusterTilePattern : index_t
{
    // Contiguous block assignment - each cluster processes a rectangular region
    ContiguousBlock = 0,

    // Interleaved along both M and N dimensions - tiles strided in both directions
    InterleavedBoth = 1,

    // Interleaved along M dimension - tiles strided along M, contiguous along N
    InterleavedM = 2
};

/**
 * @brief Class mapping 2D block index into 2D output tile space with cluster tiling.
 *
 * @tparam BlockGemmShapeType - A class providing basic GEMM parameters.
 * @tparam Pattern  - Cluster tile mapping pattern
 *
 */
template <typename BlockGemmShapeType,
          ClusterTilePattern Pattern = ClusterTilePattern::ContiguousBlock>
struct GemmClusterTilePartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t ClusterM = BlockGemmShape::kclusterM;
    static constexpr index_t ClusterN = BlockGemmShape::kclusterN;
    static constexpr index_t ClusterK = BlockGemmShape::kclusterK;

    static_assert(ClusterK == 1, "only support ClusterK == 1");

    CK_TILE_HOST_DEVICE GemmClusterTilePartitioner() noexcept = delete;
    CK_TILE_HOST_DEVICE GemmClusterTilePartitioner(index_t M_, index_t N_) noexcept : M(M_), N(N_)
    {
    }

    /**
     * @brief Calculates the grid size (in blocks) required to cover a GEMM operation, rounding up
     * to cluster sizes.
     *
     * This function computes the number of blocks needed in the X and Y dimensions to
     * process an MxN matrix, given the per-block sizes (MPerBlock, NPerBlock) and cluster sizes
     * (ClusterM, ClusterN). The grid dimensions are rounded up to the nearest multiples of the
     * cluster sizes.
     *
     * @param M GEMM's M dimension.
     * @param N GEMM's N dimension.
     * @return dim3 The grid dimensions (GridDimXRoundUp, GridDimYRoundUp, 1).
     */
    CK_TILE_HOST_DEVICE static auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX        = integer_divide_ceil(M, MPerBlock);
        const index_t GridDimY        = integer_divide_ceil(N, NPerBlock);
        const index_t GridDimXRoundUp = integer_divide_ceil(GridDimX, ClusterM) * ClusterM;
        const index_t GridDimYRoundUp = integer_divide_ceil(GridDimY, ClusterN) * ClusterN;
        return dim3(GridDimXRoundUp, GridDimYRoundUp, 1);
    }

    /**
     * @brief Calculate number of loop iterations over K dimension for given work unit
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(uint32_t K) noexcept -> uint32_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /*
    ============================================================================
    Cluster Tile Patterns: Mapping Block Index to Output Tile Coordinates
    ============================================================================

    PURPOSE:
    These patterns determine how 2D block indices (blockIdx.x, blockIdx.y) are
    mapped to output tile coordinates (tile_m, tile_n) in a GEMM operation when
    using cluster launch.


    EXAMPLE CONFIGURATION:
    - Cluster dimensions: ClusterM = 2, ClusterN = 2 (2x2 cluster)
    - Grid dimensions:    GridM = 6, GridN = 4 (6x4 output tiles)
    - Number of clusters: (6/2) x (4/2) = 3 x 2 = 6 clusters
    - Blocks per cluster: 2 x 2 = 4 blocks

    The tables below show which BLOCK (identified by its flattened cluster_id) processes
    each output TILE position (tile_m, tile_n). Values 0-5 represent the 6
    different clusters in the grid.

    ===========================================================================
    Pattern::ContiguousBlock (ClusterTilePattern::ContiguousBlock)
    ===========================================================================

    DESCRIPTION:
    Tiles are assigned in CONTIGUOUS blocks within each cluster. Each cluster
    processes a rectangular region of output tiles. This is the simplest pattern.

    TILE ASSIGNMENT (each cell shows which cluster processes that tile):

             N-> 0    1    2    3
           +------------------------+
      M  0 | |  0  |  0  |  3  |  3  |
           | +--------------------+
      |  1 | |  0  |  0  |  3  |  3  |
           | +--------------------+
      v  2 | |  1  |  1  |  4  |  4  |
           | +--------------------+
         3 | |  1  |  1  |  4  |  4  |
           | +--------------------+
         4 | |  2  |  2  |  5  |  5  |
           | +--------------------+
         5 | |  2  |  2  |  5  |  5  |
           +------------------------+

    CLUSTER LAYOUT:
    - Cluster 0: tiles (0,0), (0,1), (1,0), (1,1) - Top-left block
    - Cluster 1: tiles (2,0), (2,1), (3,0), (3,1) - Middle-left block
    - Cluster 2: tiles (4,0), (4,1), (5,0), (5,1) - Bottom-left block
    - Cluster 3: tiles (0,2), (0,3), (1,2), (1,3) - Top-right block
    - Cluster 4: tiles (2,2), (2,3), (3,2), (3,3) - Middle-right block
    - Cluster 5: tiles (4,2), (4,3), (5,2), (5,3) - Bottom-right block

    ===========================================================================
    Pattern::InterleavedBoth (ClusterTilePattern::InterleavedBoth)
    ===========================================================================

    DESCRIPTION:
    Tiles are INTERLEAVED along both M and N dimensions. Within each cluster,
    tiles are strided in both directions, creating a distributed access pattern.

    TILE ASSIGNMENT (interleaved along both M and N):

             N-> 0    1    2    3
           +------------------------+
      M  0 | |  0  |  3  |  0  |  3  |
           | +--------------------+
      |  1 | |  1  |  4  |  1  |  4  |
           | +--------------------+
      v  2 | |  2  |  5  |  2  |  5  |
           | +--------------------+
         3 | |  0  |  3  |  0  |  3  |
           | +--------------------+
         4 | |  1  |  4  |  1  |  4  |
           | +--------------------+
         5 | |  2  |  5  |  2  |  5  |
           +------------------------+

    CLUSTER LAYOUT:
    - Cluster 0: tiles (0,0), (0,2), (3,0), (3,2) - Strided along M and N
    - Cluster 1: tiles (1,0), (1,2), (4,0), (4,2) - Strided along M and N
    - Cluster 2: tiles (2,0), (2,2), (5,0), (5,2) - Strided along M and N
    - Cluster 3: tiles (0,1), (0,3), (3,1), (3,3) - Strided along M and N
    - Cluster 4: tiles (1,1), (1,3), (4,1), (4,3) - Strided along M and N
    - Cluster 5: tiles (2,1), (2,3), (5,1), (5,3) - Strided along M and N

    ===========================================================================
    Pattern::InterleavedM (ClusterTilePattern::InterleavedM)
    ===========================================================================

    DESCRIPTION:
    Tiles are INTERLEAVED along the M dimension while contiguous along N.
    Within each cluster, tiles are strided in M but adjacent in N dimension.

    TILE ASSIGNMENT (interleaved along M, contiguous along N):

             N-> 0    1    2    3
           +------------------------+
      M  0 | |  0  |  0  |  3  |  3  |
           | +--------------------+
      |  1 | |  1  |  1  |  4  |  4  |
           | +--------------------+
      v  2 | |  2  |  2  |  5  |  5  |
           | +--------------------+
         3 | |  0  |  0  |  3  |  3  |
           | +--------------------+
         4 | |  1  |  1  |  4  |  4  |
           | +--------------------+
         5 | |  2  |  2  |  5  |  5  |
           +------------------------+

    CLUSTER LAYOUT:
    - Cluster 0: tiles (0,0), (0,1), (3,0), (3,1) - Strided along M, contiguous N
    - Cluster 1: tiles (1,0), (1,1), (4,0), (4,1) - Strided along M, contiguous N
    - Cluster 2: tiles (2,0), (2,1), (5,0), (5,1) - Strided along M, contiguous N
    - Cluster 3: tiles (0,2), (0,3), (3,2), (3,3) - Strided along M, contiguous N
    - Cluster 4: tiles (1,2), (1,3), (4,2), (4,3) - Strided along M, contiguous N
    - Cluster 5: tiles (2,2), (2,3), (5,2), (5,3) - Strided along M, contiguous N

    IMPLEMENTATION NOTE:
    The actual mapping is implemented in GetOutputTileIndex() method, which
    uses cluster_id and block_offset to compute final tile coordinates based
    on the selected pattern.
    */
    CK_TILE_DEVICE static auto
    GetOutputTileIndex(index_t blockIdx, index_t blockIdy) noexcept -> const tuple<index_t, index_t>
    {
        if constexpr(Pattern == ClusterTilePattern::ContiguousBlock)
        {
            const index_t iM = amd_wave_read_first_lane(blockIdx);
            const index_t iN = amd_wave_read_first_lane(blockIdy);
            return make_tuple(iM, iN);
        }
        else
        {
            const index_t iM = amd_wave_read_first_lane(blockIdx);
            const index_t iN = amd_wave_read_first_lane(blockIdy);

            const index_t cluster_m = get_cluster_id_x();
            const index_t cluster_n = get_cluster_id_y();

            const index_t cluster_offset_m = iM % ClusterM;
            const index_t cluster_offset_n = iN % ClusterN;

            const index_t cluster_m_num = amd_wave_read_first_lane(gridDim.x / ClusterM);
            const index_t cluster_n_num = amd_wave_read_first_lane(gridDim.y / ClusterN);

            if constexpr(Pattern == ClusterTilePattern::InterleavedBoth)
            {
                const index_t tileM = cluster_m + cluster_m_num * cluster_offset_m;
                const index_t tileN = cluster_n + cluster_n_num * cluster_offset_n;
                return make_tuple(tileM, tileN);
            }
            else // InterleavedM
            {
                const index_t tileM = cluster_m + cluster_m_num * cluster_offset_m;
                const index_t tileN = cluster_n * ClusterN + cluster_offset_n;
                return make_tuple(tileM, tileN);
            }
        }
    }

    CK_TILE_DEVICE static auto
    GetOutputTileIndex(index_t blockId) noexcept -> const tuple<index_t, index_t>
    {
        const index_t iM = amd_wave_read_first_lane(blockId % gridDim.x);
        const index_t iN = amd_wave_read_first_lane(blockId / gridDim.x);
        return GetOutputTileIndex(iM, iN);
    }

    private:
    index_t M, N;
};
} // namespace ck_tile
