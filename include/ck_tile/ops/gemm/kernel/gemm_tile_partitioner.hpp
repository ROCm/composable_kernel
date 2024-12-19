// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

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

    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K)
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

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N)
    {
        index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        return dim3(GridDimX * GridDimY, 1, 1);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetNBlock(index_t N)
    {
        return integer_divide_ceil(N, NPerBlock);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K)
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    CK_TILE_DEVICE auto operator()(index_t blockOffset, index_t NBlockSize)
    {
        index_t iM = __builtin_amdgcn_readfirstlane((blockIdx.x - blockOffset) /
                                                    GetNBlock(NBlockSize) * MPerBlock);
        index_t iN = __builtin_amdgcn_readfirstlane((blockIdx.x - blockOffset) %
                                                    GetNBlock(NBlockSize) * NPerBlock);
        return make_tuple(iM, iN);
    }
};
} // namespace ck_tile
