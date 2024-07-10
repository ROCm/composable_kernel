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
                                                ck_tile::index_t seqlen_knew,
                                                ck_tile::index_t hdim_v)
    {
        assert(ck_tile::integer_divide_ceil(hdim_v, kTileSizeD) == 1);
#ifdef NDEBUG
        ignore = hdim_v;
#endif

        // TODO: this may need tuning
        return dim3(ck_tile::integer_divide_ceil(seqlen_knew, kTileSizeSk), nhead, batch_size);
    }

    CK_TILE_DEVICE auto operator()(ck_tile::index_t /*seqlen_q*/, ck_tile::index_t /*hdim_v*/)
    {
        // const index_t num_tile_n1 = ck_tile::integer_divide_ceil(hdim_v, kN1);

        const index_t i_block = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };
        (void)f;
        // const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

        return ck_tile::make_tuple(i_block, i_nhead, i_batch);
    }
};

} // namespace ck_tile
