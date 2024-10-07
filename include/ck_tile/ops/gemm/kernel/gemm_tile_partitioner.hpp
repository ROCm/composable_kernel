// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
template <typename BlockGemmShape_>
struct GemmTilePartitioner
{
    using BlockGemmShape = ck_tile::remove_cvref_t<BlockGemmShape_>;

    static constexpr ck_tile::index_t kM = BlockGemmShape::kM;
    static constexpr ck_tile::index_t kN = BlockGemmShape::kN;
    static constexpr ck_tile::index_t kK = BlockGemmShape::kK;

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t batch_size)
    {
        ck_tile::index_t GridDimX = (M + kM - 1) / kM;
        ck_tile::index_t GridDimY = (N + kN - 1) / kN;
        ck_tile::index_t GridDimZ = batch_size;
        return dim3(GridDimX, GridDimY, GridDimZ);
    }

    CK_TILE_DEVICE auto operator()()
    {
        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx.x * kM);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdx.y * kN);
        return ck_tile::make_tuple(iM, iN);
    }
};
} // namespace ck_tile
