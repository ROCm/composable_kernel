// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
template <typename CrossReduceShape_>
struct CrossReducePartitioner
{
    using CrossReduceShape = remove_cvref_t<CrossReduceShape_>;

    static constexpr index_t kM = CrossReduceShape::Block_M;
    static constexpr index_t kN = CrossReduceShape::Block_N;

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N)
    {
        index_t GridDimX = (M + kM - 1) / kM;
        index_t GridDimY = (N + kN - 1) / kN;
        return dim3(GridDimX, GridDimY, 1);
    }

    CK_TILE_DEVICE auto operator()() {
        const index_t i_M = __builtin_amdgcn_readfirstlane(blockIdx.x * kM);
        return i_M;
    }
};
} // namespace ck_tile
