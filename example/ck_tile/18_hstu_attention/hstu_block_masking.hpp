// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kUseCausal, bool kUseLocal>
struct HstuBlockMasking
{
    static constexpr bool IsMasking = (kUseCausal || kUseLocal);

    int contextual_seqlen;
    int max_uih_len;

    int max_attn_len;
    int min_full_attn_seqlen;

    CK_TILE_HOST_DEVICE HstuBlockMasking(int seqlen_,
                                         int contextual_seqlen_,
                                         int num_target,
                                         int max_attn_len_,
                                         int min_full_attn_seqlen_)
    {
        max_uih_len       = seqlen_;
        contextual_seqlen = contextual_seqlen_;

        max_attn_len         = max_attn_len_;
        min_full_attn_seqlen = min_full_attn_seqlen_;

        max_uih_len -= contextual_seqlen > 0 ? contextual_seqlen - 1 : 0;
        max_uih_len -= num_target;
    };

    CK_TILE_HOST_DEVICE HstuBlockMasking(int seqlen_, int contextual_seqlen_, int num_target)
    {
        max_uih_len       = seqlen_;
        contextual_seqlen = contextual_seqlen_;

        max_uih_len -= contextual_seqlen > 0 ? contextual_seqlen - 1 : 0;
        max_uih_len -= num_target;
    };

    // to get the loop length along X axis, return index:[start, end), end-start=length
    // use this if need loop over X axis tile by tile (eg. seqlen_k loop-over)
    // i_y is the start offset of the current tile along the seqlen_q dimension
    template <index_t YTile, index_t XTile>
    CK_TILE_HOST_DEVICE constexpr auto
    GetTileRangeAlongX(index_t i_y, number<YTile>, number<XTile>) const
    {
        if constexpr(!IsMasking)
        {
            return ck_tile::make_tuple(0, max_uih_len);
        }
        else
        {
            if(i_y < contextual_seqlen)
                return ck_tile::make_tuple(0, max_uih_len);

            if constexpr(kUseCausal && !kUseLocal)
            {
                index_t x_end =
                    min(i_y + YTile, max_uih_len); // for lower-triangular masking, x <= y

                return ck_tile::make_tuple(0, x_end);
            }
            else if constexpr(!kUseCausal && kUseLocal)
            {
                if(min_full_attn_seqlen > 0 && i_y + YTile > max_uih_len - min_full_attn_seqlen)
                {
                    return ck_tile::make_tuple(0, max_uih_len);
                }
                else
                {
                    index_t x_start = max(0, i_y - max_attn_len);
                    index_t x_end   = i_y + YTile + max_attn_len;

                    return ck_tile::make_tuple(x_start - x_start % XTile, x_end);
                };
            }
            else // kUseCausal && kUseLocal
            {
                if(min_full_attn_seqlen > 0 && i_y + YTile > max_uih_len - min_full_attn_seqlen)
                {
                    return ck_tile::make_tuple(0, max_uih_len);
                }
                else
                {
                    index_t x_end = i_y + YTile + max_attn_len;

                    return ck_tile::make_tuple(0, x_end);
                };
            };
        };
    }

    CK_TILE_HOST_DEVICE constexpr bool IsTokenPairInsideMask(int row, int col)
    {
        if(row >= max_uih_len || col >= max_uih_len)
            return false;

        if(row < contextual_seqlen)
            return true;

        if constexpr(IsMasking)
        {
            bool result = false;
            if constexpr(kUseLocal)
            {
                if constexpr(kUseCausal)
                    result = (row >= col) && (row - col <= max_attn_len);
                else
                    result = abs(row - col) <= max_attn_len;

                if(min_full_attn_seqlen > 0)
                    result = (row >= max_uih_len - min_full_attn_seqlen);
            }
            else
            {
                result = (row >= col);
            };

            return result;
        }

        return true;
    };
};

} // namespace ck_tile
