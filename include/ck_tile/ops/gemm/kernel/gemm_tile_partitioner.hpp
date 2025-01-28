// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/** @brief Struct representing 2D block index mapping into 3D output tile space. */
template <typename BlockGemmShapeType>
struct GemmTile2DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    /** @brief Returns 3D grid size. */
    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t batch_size) noexcept(
        noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        const index_t GridDimZ = batch_size;
        return dim3(GridDimX, GridDimY, GridDimZ);
    }

    /**
     * @brief Returns the number of loops.
     * @param [in] K is dimension
     */
    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief The function returns 2D output tile space.
     * @param [in] blockIdx is blockIdx.x
     * @param [in] blockIdy is blockIdx.y
     * @return Returns the output tile indexes.
     */
    CK_TILE_DEVICE static constexpr auto GetOutputTileIndex(index_t blockIdx,
                                                            index_t blockIdy) noexcept
        -> const tuple<index_t, index_t>
    {
        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdy);
        return make_tuple(iM, iN);
    }
};

/**
 * @brief Struct representing 1D block index mapping into 2D output tile space.
 */
template <typename BlockGemmShapeType>
struct GemmTile1DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    /** @brief delete default ctr with no any object */
    constexpr GemmTile1DPartitioner() noexcept = delete;

    /** @brief constructs an object that does contain a N value. */
    constexpr GemmTile1DPartitioner(index_t N) noexcept { N_ = N; }

    /** @brief Returns 1D grid size. */
    CK_TILE_HOST static constexpr auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        return dim3(GridDimX * GridDimY, 1, 1);
    }

    /**
     * @brief Returns the number of blocks in N.
     * @param [in] N is dimension
     */
    CK_TILE_HOST_DEVICE static constexpr auto GetNBlock(index_t N) noexcept -> index_t
    {
        return integer_divide_ceil(N, NPerBlock);
    }

    /**
     * @brief Returns the number of loops.
     * @param [in] K is dimension
     */
    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief The function returns 2D output tile space.
     * @param [in] blockIdx is blockIdx.x - block_start.
     * */
    CK_TILE_DEVICE static constexpr auto GetOutputTileIndex(index_t blockIdx) noexcept
        -> const tuple<index_t, index_t>
    {
        const index_t NBlock = GetNBlock(N_);

        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx / NBlock);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdx - iM * NBlock);
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
template <typename PartitionerFn,
          typename = typename std::enable_if_t<HasFnOneArgImpl<PartitionerFn>{}>>
struct OffsettedTile1DPartitioner
{
    /**
     * @brief The function subtracts the block's start (offset) from 1D raw-indexes.
     * @param [in] block_start is `blockIdx.x - block_start`.
     * @return Returns a `tuple` [Im, In] shifted index, used to shift 1d-tile index.
     */
    [[nodiscard]] CK_TILE_DEVICE static constexpr auto GetOffsetedTileIndex(index_t block_start,
                                                                            index_t N) noexcept
        -> const tuple<index_t, index_t>
    {
        const auto [iM, iN] = PartitionerFn(N).GetOutputTileIndex(blockIdx.x - block_start);
        return make_tuple(iM, iN);
    }
};

/**
 * @brief Class mapping 1D block index into 2D output tile space.
 *
 * @note It groups spatially workgroups in order to better utilize caches.
 *       It is using grouped Rows of column-vectors WGP pattern. It's optimized
 *       for gfx94x-like multiple-die chip.
 */
template <typename BlockGemmShapeType, index_t GroupNum, index_t M01>
struct GemmSpatiallyLocalTilePartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE constexpr GemmSpatiallyLocalTilePartitioner() noexcept = delete;
    CK_TILE_HOST_DEVICE constexpr GemmSpatiallyLocalTilePartitioner(index_t M_, index_t N_) noexcept
        : M(M_), N(N_)
    {
    }

    /** @brief Returns 1D grid size. */
    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t batch_size) noexcept(
        noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX = integer_divide_ceil(M, MPerBlock);
        const index_t GridDimY = integer_divide_ceil(N, NPerBlock);
        return dim3(GridDimX * GridDimY, 1, batch_size);
    }

    /**
     * @brief Returns the number of loop over K dimension.
     * @param [in] K    GEMM's K dimension.
     */
    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief The function returns 2D output tile space.
     * @param [in] block_1d_id - 1D linear workgroup index.
     * */
    CK_TILE_DEVICE constexpr auto GetOutputTileIndex(index_t block_1d_id) noexcept
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

} // namespace ck_tile
