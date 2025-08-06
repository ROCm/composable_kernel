// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct BatchedTransposeCommonPolicy
{
    CK_TILE_DEVICE static constexpr auto TileAccessPattern =
        tile_distribution_pattern::thread_raked;

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeInputDistribution()
    {
        constexpr index_t BlockSize         = Problem::kBlockSize;
        constexpr index_t LeadDimPerBlock   = Problem::kMPerBlock;
        constexpr index_t SecondDimPerBlock = Problem::kNPerBlock;

        constexpr index_t kVectorSize = Problem::VectorSizeOutput;

        using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                      SecondDimPerBlock,
                                                                      LeadDimPerBlock,
                                                                      kVectorSize,
                                                                      TileAccessPattern>;
        return TileEncodingPattern::Make2DStaticTileDistribution();
    }
};

} // namespace ck_tile
