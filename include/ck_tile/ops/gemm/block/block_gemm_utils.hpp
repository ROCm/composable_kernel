// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename AccWarpDescEnc,
          typename BlockTile, // seq<M, N>
          typename BlockWarps,
          typename WarpTile>
CK_TILE_DEVICE_HOST constexpr auto make_block_gemm_acc_enc()
{
    constexpr index_t Block_M = BlockTile::at(number<0>{});
    constexpr index_t Block_N = BlockTile::at(number<1>{});

    constexpr index_t BlockWarps_M = BlockWarps::at(number<0>{});
    constexpr index_t BlockWarps_N = BlockWarps::at(number<1>{});

    constexpr index_t Warp_M = WarpTile::at(number<0>{});
    constexpr index_t Warp_N = WarpTile::at(number<1>{});

    constexpr index_t Repeat_M = Block_M / (BlockWarps_M * Warp_M);
    constexpr index_t Repeat_N = Block_N / (BlockWarps_N * Warp_N);

    constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Repeat_M, BlockWarps_M>, sequence<Repeat_N, BlockWarps_N>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2>,
        sequence<0, 0>>{};

    constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
        c_block_outer_dstr_encoding, AccWarpDescEnc{});
    return c_block_dstr_encode;
}

} // namespace ck_tile
