// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Default policy for batchnorm forward pipeline
// Defines tile distributions and helper functions
struct BatchnormFwdPipelineDefaultPolicy
{
    // Tile distribution for input data (following layernorm2d pattern exactly)
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
                      sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }

    // Tile distribution for gamma/beta parameters (following layernorm2d pattern)
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeGammaBetaBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<S::WarpPerBlock_M, S::ThreadPerWarp_M>,
                tuple<sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<0, 1>, sequence<0, 1>>,
                tuple<sequence<0, 1>, sequence<1, 2>>,
                sequence<1, 1>,
                sequence<0, 3>>{});
    }

    // Simple 1D tile distribution for transformed [N×H×W] windows
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto Make1DBlockTileDistribution()
    {
        // For merged 1D data, use simple pass-through distribution
        // All threads collaborate on the Block_N elements
        using S = typename Problem::BlockShape;
        
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<0>>,
                tuple<sequence<0>>,
                sequence<1>,
                sequence<0>>{});
    }

    // Calculate shared memory size
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // For POC, use BlockWelford's smem requirement
        using ComputeDataType = typename Problem::ComputeDataType;
        constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;
        
        return BlockWelford<ComputeDataType>::template GetSmemSize<index_t, kBlockSize>();
    }
};

} // namespace ck_tile
