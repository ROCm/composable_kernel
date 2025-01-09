// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/tile_distribution_encoding.hpp"

namespace ck_tile {

/**
 * @brief Enumeration describing tile distribution patterns.
 *
 */
enum struct tile_distribution_pattern
{
    /**
     * @brief Thread raked pattern.
     *
     */
    thread_raked,
    /**
     * @brief Warp raked pattern.
     *
     */
    warp_raked,
    /**
     * @brief Block raked pattern - aka linear.
     *
     */
    block_raked,
    // TODO pattern taking into account MFMA attributes:
    // block_fmha_pipeline_qx_ks_vs_custom_policy.hpp::51 MakeQDramTileDistribution()
};

struct TileDistributionEcodingPattern
{
};

/**
 * @brief Class creating 2D static tile distribution with different load/store patterns.
 *
 * @note We always assume that Tile is YPerTile x XPerTile where X dim (rightmost)
 *       is contiguous and we can do vector load on this dimension.
 *
 * @tparam BlockSize    Number of threads in a workgroup.
 * @tparam YPerTile    The tile size of outer/leftmost dimension.
 * @tparam XPerTile    The tile size of inner/rightmost dimension (contiguous).
 * @tparam VecSize      The vector access size.
 * @tparam DistributionPattern The enumeration describing used access pattern.
 */
template <index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize,
          tile_distribution_pattern DistributionPattern>
struct TileDistributionEncodingPattern2D : public TileDistributionEcodingPattern
{
};

// Thread raked
template <index_t BlockSize, index_t YPerTile, index_t XPerTile, index_t VecSize>
struct TileDistributionEncodingPattern2D<BlockSize,
                                         YPerTile,
                                         XPerTile,
                                         VecSize,
                                         tile_distribution_pattern::thread_raked>
    : public TileDistributionEcodingPattern
{

    CK_TILE_HOST_DEVICE static constexpr auto Make2DStaticTileDistribution()
    {
        // TODO: make pattern where below condition does not need to hold - GGemmMultiDSplitk!
        static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
        constexpr index_t warp_size = get_warp_size();
        constexpr index_t num_warps = BlockSize / get_warp_size();
        constexpr index_t X1        = VecSize;
        constexpr index_t X0        = XPerTile / X1; // # of threads in X dim

        // # of rows in Y dim accessed by single wavefront in one iteration
        constexpr index_t Y1 = warp_size / X0;
        static_assert(X0 * Y1 == warp_size, "X0 * Y1 must cover whole wavefront!");

        constexpr index_t Y0 = num_warps;
        //  YPerWarp = YPerTile / Y0;
        //  Y2 = YPerWarp / Y1;
        constexpr index_t Y2 = YPerTile / (Y1 * Y0); // # of iters within wavefront

        static_assert(X0 * Y1 * Y0 == BlockSize, "X0 * warp_ys * Y0 must cover whole workgroup!");
        static_assert(Y0 * Y1 * Y2 == YPerTile, "Y0, Y1, Y2 must cover whole YPerTile");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<2, 1>>{});
    }
};

// Warp raked
template <index_t BlockSize, index_t YPerTile, index_t XPerTile, index_t VecSize>
struct TileDistributionEncodingPattern2D<BlockSize,
                                         YPerTile,
                                         XPerTile,
                                         VecSize,
                                         tile_distribution_pattern::warp_raked>
    : public TileDistributionEcodingPattern
{
    CK_TILE_HOST_DEVICE static constexpr auto Make2DStaticTileDistribution()
    {
        static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
        constexpr index_t warp_size = get_warp_size();
        constexpr index_t num_warps = BlockSize / get_warp_size();
        constexpr index_t X1        = VecSize;
        constexpr index_t X0        = XPerTile / X1; // # of threads in X dim

        constexpr index_t Y2 = warp_size / X0; // # of rows in Y dim to cover whole wavefront
        static_assert(X0 * Y2 == warp_size, "X0 * Y2 must cover whole wavefront!");

        constexpr index_t Y0 = num_warps;
        static_assert(X0 * Y2 * Y0 == BlockSize, "X0 * Y2 * Y1 must cover whole workgroup!");

        constexpr index_t Y1 = YPerTile / (Y2 * Y0); // # of iters within wavefront
        static_assert(Y0 * Y1 * Y2 == YPerTile, "Y0, Y1, Y2 must cover whole YPerTile");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 1>>{});
    }
};

// Block raked
template <index_t BlockSize, index_t YPerTile, index_t XPerTile, index_t VecSize>
struct TileDistributionEncodingPattern2D<BlockSize,
                                         YPerTile,
                                         XPerTile,
                                         VecSize,
                                         tile_distribution_pattern::block_raked>
    : public TileDistributionEcodingPattern
{

    CK_TILE_HOST_DEVICE static constexpr auto Make2DStaticTileDistribution()
    {
        // TODO: make pattern where below condition does not need to hold - GGemmMultiDSplitk!
        static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
        constexpr index_t warp_size = get_warp_size();
        constexpr index_t num_warps = BlockSize / get_warp_size();
        constexpr index_t X1        = VecSize;
        constexpr index_t X0        = XPerTile / X1;  // # of threads in X dim
        constexpr index_t Y2        = warp_size / X0; // # of rows in Y dim to cover whole wavefront
        static_assert(X0 * Y2 == warp_size, "X0 * Y2 must cover whole wavefront!");
        constexpr index_t Y1 = num_warps;
        static_assert(X0 * Y2 * Y1 == BlockSize, "X0 * Y2 * Y1 must cover whole workgroup!");
        constexpr index_t Y0 = YPerTile / (Y2 * Y1);
        static_assert(Y0 * Y1 * Y2 == YPerTile, "Y0, Y1, Y2 must cover whole YPerTile");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }
};

} // namespace ck_tile
