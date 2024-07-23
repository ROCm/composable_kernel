// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <index_t kTileSizeS_, index_t kTileSizeSk_, index_t kTileSizeD_, index_t kTileSizeDv_>
struct FmhaFwdAppendKVTilePartitioner
{
    static constexpr ck_tile::index_t kTileSizeS  = kTileSizeS_;
    static constexpr ck_tile::index_t kTileSizeSk = kTileSizeSk_;
    static constexpr ck_tile::index_t kTileSizeD  = kTileSizeD_;
    static constexpr ck_tile::index_t kTileSizeDv = kTileSizeDv_;

    static_assert(kTileSizeD == kTileSizeDv);

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size,
                                                ck_tile::index_t nhead,
                                                ck_tile::index_t seqlen_q,
                                                ck_tile::index_t seqlen_knew)
    {
        // TODO: this may need tuning
        return dim3(std::max(ck_tile::integer_divide_ceil(seqlen_q, kTileSizeS),
                             ck_tile::integer_divide_ceil(seqlen_knew, kTileSizeSk)),
                    nhead,
                    batch_size);
    }

    CK_TILE_DEVICE auto operator()()
    {
        const index_t i_tile  = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        return ck_tile::make_tuple(i_tile, i_nhead, i_batch);
    }
};

} // namespace ck_tile
