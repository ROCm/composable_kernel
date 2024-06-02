// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockFmhaShape_>
struct FmhaFwdSplitKVCombineTilePartitioner
{
    using BlockFmhaShape = ck_tile::remove_cvref_t<BlockFmhaShape_>;

    static constexpr ck_tile::index_t kM0 = BlockFmhaShape::kM0;
    static constexpr ck_tile::index_t kN1 = BlockFmhaShape::kN0;
    // constexpr static ck_tile::index_t kBlockM = kN1 % 128 == 0 ? 4 : (kN1 % 64 == 0 ? 8 : 16);

    __host__ static constexpr auto
    GridSize(ck_tile::index_t batch_size, ck_tile::index_t nhead, ck_tile::index_t max_seqlen_q)
    {
        return dim3(ck_tile::integer_divide_ceil(max_seqlen_q, kM0), nhead, batch_size);
    }

    CK_TILE_DEVICE auto operator()(ck_tile::index_t /*seqlen_q*/, ck_tile::index_t /*hdim_v*/)
    {
        const index_t i_tile_m = blockIdx.x;
        const index_t i_nhead  = blockIdx.y;
        const index_t i_batch  = blockIdx.z;

        return ck_tile::make_tuple(i_tile_m, i_nhead, i_batch);
    }
};

} // namespace ck_tile
