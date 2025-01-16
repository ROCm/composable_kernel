// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @brief An internal class dedicated to 2D tiles.
template <typename BlockGemmShapeType>
struct GemmTile2DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    /// @brief Returns 2D-grid size.
    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t batch_size) noexcept(
        noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        const index_t GridDimZ = batch_size;
        return dim3(GridDimX, GridDimY, GridDimZ);
    }

    /// @brief Returns the number of loops from the the ceiling of M divided by KPerBlock.
    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /// @brief Returns the tile raw indexes. Call with blockIdx and blockIdy, where:
    // blockIdx = blockId.x and blockIdy blockIdx.y.
    CK_TILE_DEVICE static constexpr auto GetOutputTileIndex(index_t blockIdx,
                                                            index_t blockIdy) noexcept
        -> const tuple<index_t, index_t>
    {
        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdy);
        return make_tuple(iM, iN);
    }
};

/// @brief An internal class dedicated to 1D tiles.
template <typename BlockGemmShapeType>
struct GemmTile1DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    /// @brief Returns 1D-grid size.
    CK_TILE_HOST static constexpr auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        return dim3(GridDimX * GridDimY, 1, 1);
    }

    /// @brief Returns the value of the ceiling of N divided by NPerBlock
    CK_TILE_HOST_DEVICE static constexpr auto GetNBlock(index_t N) noexcept -> index_t
    {
        return integer_divide_ceil(N, NPerBlock);
    }

    /// @brief Returns the number of loops from the the ceiling of M divided by KPerBlock.
    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /// @brief Set the N block size.
    CK_TILE_DEVICE static constexpr auto SetNBlock(index_t N) noexcept -> void { _NBlockSize = N; }

    /// @brief Returns the tile raw index. Call with blockIdx, which is offset (blockIdx.x -
    /// block_start). Note: The `SetNBlock` must be called before calling this function.
    CK_TILE_DEVICE static constexpr auto GetOutputTileIndex(index_t blockIdx) noexcept
        -> const tuple<index_t, index_t>
    {
        const index_t NBlock = GetNBlock(_NBlockSize);

        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx / NBlock);
        const index_t iN = __builtin_amdgcn_readfirstlane(modulo(blockIdx, NBlock));
        return make_tuple(iM, iN);
    }

    private:
    CK_TILE_DEVICE static index_t _NBlockSize;

    /// @brief Return modulo value.
    [[nodiscard]] CK_TILE_DEVICE static auto constexpr modulo(index_t input, index_t ceil) noexcept
        -> index_t
    {
        return input >= ceil ? input - (input / ceil) * ceil : input;
    }
};

/// @brief `GemmTile1DPartitioner::GetOutputTileIndex`'s std::false specialization,
/// checking expression validity in-place for ill-formed.
template <typename, typename = void>
struct HasFnOneArgImpl : std::false_type
{
};

/// @brief `GemmTile1DPartitioner::GetOutputTileIndex`'s std::true specialization,
/// checking expression validity in-place for well-formed.
/// Note: `1` - a constant value indicating the number of parameters in the function.
template <typename T>
struct HasFnOneArgImpl<T, std::void_t<decltype(std::declval<T>().GetOutputTileIndex(1))>>
    : std::true_type
{
};

/// @brief Struct `OffsetCallculation1DPartitioner` class.
/// used to calculate offseted tile indexes.
/// Note: The struct supports the 1D-Partitioner mechanism, enable-if `GetOutputTileIndex`-fn
/// is std::true_type when `GetOutputTileIndex`-fn is well-formed, otherwise std::false_type.
template <typename PartitionerFn,
          typename = typename std::enable_if_t<HasFnOneArgImpl<PartitionerFn>{}>>
struct OffsetCallculation1DPartitioner
{
    /// @brief  Returns a `tuple` [Im, In] shifted index, used to shift 1d-tile index.
    //  Note: The function subtracts the block's start (offset) from 1D raw-indexes.
    [[nodiscard]] CK_TILE_DEVICE static constexpr auto
    GetOffsetedTileIndex(index_t block_start) noexcept -> const tuple<index_t, index_t>
    {
        const auto [iM, iN] = PartitionerFn::GetOutputTileIndex(blockIdx.x - block_start);
        return make_tuple(iM, iN);
    }
};
} // namespace ck_tile
