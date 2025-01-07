// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockGemmShape_>
struct GemmTilePartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t batch_size)
    {
        index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        index_t GridDimZ = batch_size;
        return dim3(GridDimX, GridDimY, GridDimZ);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K) -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    CK_TILE_DEVICE auto operator()()
    {
        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx.x * MPerBlock);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdx.y * NPerBlock);
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
        index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
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

    CK_TILE_DEVICE auto constexpr
    operator()(index_t blockIdx, index_t NBlockSize) noexcept(noexcept(GetNBlock(NBlockSize) != 0))
        -> const tuple<index_t, index_t>
    {
        const index_t NBlock = GetNBlock(NBlockSize);

        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx / NBlock);
        const index_t iN = __builtin_amdgcn_readfirstlane(modulo(blockIdx, NBlock));

        return make_tuple(iM, iN);
    }

    private:
    CK_TILE_DEVICE auto constexpr modulo(index_t input, index_t ceil) noexcept(noexcept(ceil != 0))
        -> index_t
    {
        return input >= ceil ? input - (input / ceil) * ceil : input;
    }
};

template <typename PartitionerFn>
struct InvokeOffsetCallculationFor1DPartitioner
{
    CK_TILE_DEVICE constexpr auto operator()(index_t block_start, index_t NBlockSize)
        -> const tuple<index_t, index_t>
    {
        const auto [iM, iN]       = PartitionerFn{}(blockIdx.x - block_start, NBlockSize);
        const auto iM_to_img_corr = iM * PartitionerFn::MPerBlock;
        const auto iN_to_img_corr = iN * PartitionerFn::NPerBlock;

        const index_t i_m = __builtin_amdgcn_readfirstlane(iM_to_img_corr);
        const index_t i_n = __builtin_amdgcn_readfirstlane(iN_to_img_corr);

        return make_tuple(i_m, i_n);
    }
};
} // namespace ck_tile
