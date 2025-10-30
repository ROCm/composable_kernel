// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "batched_transpose_common_policy.hpp"

namespace ck_tile {

struct BatchedTransposePolicy : public BatchedTransposeCommonPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOutputDistribution()
    {
        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t MPerBlock   = Problem::kMPerBlock;
        constexpr index_t NPerBlock   = Problem::kNPerBlock;
        constexpr index_t VecLoadSize = Problem::VectorSizeOutput;

        using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                      NPerBlock,
                                                                      MPerBlock,
                                                                      VecLoadSize,
                                                                      TileAccessPattern>;
        return TileEncodingPattern::MakeShuffled2DStaticTileDistribution();
    }
};
} // namespace ck_tile
