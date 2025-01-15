// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockGemmShape_>
struct GemmTile2DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t batch_size) noexcept(
        noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        const index_t GridDimZ = batch_size;
        return dim3(GridDimX, GridDimY, GridDimZ);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K) -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    CK_TILE_DEVICE static constexpr auto GetOutputTileIndex(index_t blockIdx,
                                                            index_t blockIdy) noexcept
        -> const tuple<index_t, index_t>
    {
        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdy);
        return make_tuple(iM, iN);
    }
};

template <typename BlockGemmShape_>
struct GemmTile1DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST static constexpr auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        return dim3(GridDimX * GridDimY, 1, 1);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetNBlock(index_t N) -> index_t
    {
        return integer_divide_ceil(N, NPerBlock);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K) -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    CK_TILE_DEVICE static constexpr auto SetNBlock(index_t N) noexcept -> void { _NBlockSize = N; }

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

    CK_TILE_DEVICE static auto constexpr modulo(index_t input, index_t ceil) -> index_t
    {
        return input >= ceil ? input - (input / ceil) * ceil : input;
    }
};

template <typename, typename = void>
struct has_1_arg_fn_impl : std::false_type
{
};

template <typename T>
struct has_1_arg_fn_impl<T, std::void_t<decltype(std::declval<T>().GetOutputTileIndex(1))>>
    : std::true_type
{
};

template <typename PartitionerFn,
          typename = typename std::enable_if_t<has_1_arg_fn_impl<PartitionerFn>{}>>
struct OffsetCallculation1DPartitioner
{
    CK_TILE_DEVICE static constexpr auto GetOffsetedTileIndex(index_t block_start)
        -> const tuple<index_t, index_t>
    {
        const auto [iM, iN] = PartitionerFn::GetOutputTileIndex(blockIdx.x - block_start);
        return make_tuple(iM, iN);
    }
};
} // namespace ck_tile
