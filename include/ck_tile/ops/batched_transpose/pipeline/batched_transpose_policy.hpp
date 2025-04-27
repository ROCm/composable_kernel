// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/softmax.hpp"
#include "ck_tile/ops/topk.hpp"

namespace ck_tile {

struct BatchedTransposePolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeInputDistribution()
    {
        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t MPerBlock   = Problem::kMPerBlock;
        constexpr index_t NPerBlock   = Problem::kNPerBlock;
        constexpr index_t VecLoadSize = Problem::VectorSizeInput;
        using TileEncodingPattern =
            TileDistributionEncodingPattern2D<BlockSize,
                                              MPerBlock,
                                              NPerBlock,
                                              VecLoadSize,
                                              tile_distribution_pattern::thread_raked>;
        return TileEncodingPattern::Make2DStaticTileDistribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOutputDistribution()
    {
        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t MPerBlock   = Problem::kMPerBlock;
        constexpr index_t NPerBlock   = Problem::kNPerBlock;
        constexpr index_t VecLoadSize = Problem::VectorSizeOutput;

        using TileEncodingPattern =
            TileDistributionEncodingPattern2D<BlockSize,
                                              NPerBlock,
                                              MPerBlock,
                                              VecLoadSize,
                                              tile_distribution_pattern::thread_raked>;
        return TileEncodingPattern::MakeShuffled2DStaticTileDistribution();
    }
};
} // namespace ck_tile
