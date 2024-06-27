// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockFmhaShape_>
struct FmhaFwdSplitKVTilePartitioner
{
    using BlockFmhaShape = ck_tile::remove_cvref_t<BlockFmhaShape_>;

    static constexpr ck_tile::index_t kM0 = BlockFmhaShape::kM0;
    static constexpr ck_tile::index_t kN0 = BlockFmhaShape::kN0;
    static constexpr ck_tile::index_t kK0 = BlockFmhaShape::kK0;
    static constexpr ck_tile::index_t kN1 = BlockFmhaShape::kN1;
    static constexpr ck_tile::index_t kK1 = BlockFmhaShape::kK1;

    __host__ static constexpr auto GridSize(ck_tile::index_t batch_size,
                                            ck_tile::index_t nhead,
                                            ck_tile::index_t seqlen_q,
                                            ck_tile::index_t hdim_v,
                                            ck_tile::index_t num_splits)
    {
        // TODO: this may need tuning
        return dim3(ck_tile::integer_divide_ceil(seqlen_q, kM0) *
                        ck_tile::integer_divide_ceil(hdim_v, kN1),
                    nhead * num_splits,
                    batch_size);
    }

    CK_TILE_DEVICE auto
    operator()(ck_tile::index_t /*seqlen_q*/, ck_tile::index_t hdim_v, ck_tile::index_t num_splits)
    {
        const index_t num_tile_n1 = ck_tile::integer_divide_ceil(hdim_v, kN1);

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        const auto [i_tile_m, i_tile_n] = f(blockIdx.x, num_tile_n1);
        const auto [i_nhead, i_split]   = f(blockIdx.y, num_splits);
        const index_t i_batch           = blockIdx.z;

        return ck_tile::make_tuple(i_tile_m, i_tile_n, i_split, i_nhead, i_batch);
    }
};

} // namespace ck_tile
