// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockGemmShape_>
struct GemmTilePartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t kM = BlockGemmShape::kM;
    static constexpr index_t kN = BlockGemmShape::kN;
    static constexpr index_t kK = BlockGemmShape::kK;

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t batch_size)
    {
        index_t GridDimX = (M + kM - 1) / kM;
        index_t GridDimY = (N + kN - 1) / kN;
        index_t GridDimZ = batch_size;
        return dim3(GridDimX, GridDimY, GridDimZ);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K) -> index_t
    {
        return integer_divide_ceil(K, kK);
    }

    CK_TILE_DEVICE auto operator()()
    {
        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx.x * kM);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdx.y * kN);
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

    CK_TILE_DEVICE auto
    operator()(index_t blockIdx, index_t NBlockSize) noexcept(noexcept(GetNBlock(NBlockSize) != 0))
        -> tuple<index_t, index_t>
    {
        const index_t NBlock = GetNBlock(NBlockSize);

        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx / NBlock);
        const index_t iN = __builtin_amdgcn_readfirstlane(fast_mod(blockIdx, NBlock));

        return make_tuple(iM, iN);
    }

    private:
    template <typename TType>
    CK_TILE_DEVICE auto fast_mod(const TType input, const TType ceil) noexcept(noexcept(ceil != 0))
        -> std::enable_if_t<std::numeric_limits<TType>::is_integer, TType>
    {
        return input >= ceil ? input - (input / ceil) * ceil : input;
    }
};

template <typename PartitionerFn>
struct InvokeOffsetCallculationFor1DPartitioner
{
    template <typename TType>
    CK_TILE_DEVICE constexpr auto operator()(TType block_start, TType NBlockSize)
        -> const std::enable_if_t<std::numeric_limits<TType>::is_integer, tuple<TType, TType>>
    {
        const auto [iM, iN]       = PartitionerFn{}(blockIdx.x - block_start, NBlockSize);
        const auto iM_to_img_corr = iM * PartitionerFn::MPerBlock;
        const auto iN_to_img_corr = iN * PartitionerFn::NPerBlock;

        const TType i_m = __builtin_amdgcn_readfirstlane(iM_to_img_corr);
        const TType i_n = __builtin_amdgcn_readfirstlane(iN_to_img_corr);

        return make_tuple(i_m, i_n);
    }
};
} // namespace ck_tile
