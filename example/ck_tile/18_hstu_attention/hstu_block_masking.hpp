// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>
#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kUseCausal>
struct HstuBlockMaskWithLocal
{
    static constexpr bool kUseLocal = true;
    static constexpr bool IsMasking = true;

    int contextual_seqlen;
    int max_uih_len;

    int max_attn_len;
    int min_full_attn_seqlen;

    CK_TILE_HOST_DEVICE HstuBlockMaskWithLocal(int contextual_seqlen_,
                                               int max_uih_len_,
                                               int max_attn_len_,
                                               int min_full_attn_seqlen_)
        : contextual_seqlen(contextual_seqlen_),
          max_uih_len(max_uih_len_),
          max_attn_len(max_attn_len_),
          min_full_attn_seqlen(min_full_attn_seqlen_){};

    // to get the loop length along X axis, return index:[start, end), end-start=length
    // use this if need loop over X axis tile by tile (eg. seqlen_k loop-over)
    // i_y is the start offset of the current tile along the seqlen_q dimension
    template <index_t YTile, index_t XTile>
    CK_TILE_HOST_DEVICE constexpr auto
    GetTileRangeAlongX(index_t i_y, number<YTile>, number<XTile>) const
    {
        if(i_y < contextual_seqlen)
            return ck_tile::make_tuple(0, max_uih_len);

        if constexpr(!kUseCausal)
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
                index_t x_start = max(0, i_y - max_attn_len);
                index_t x_end   = min(i_y + YTile, max_uih_len);

                return ck_tile::make_tuple(x_start, x_end);
            };
        };
    }

    CK_TILE_HOST constexpr bool IsTokenPairInsideMask(int row, int col)
    {
        if(row >= max_uih_len || col >= max_uih_len)
            return false;

        if(row < contextual_seqlen)
            return true;

        bool result = false;
        if constexpr(kUseCausal)
            result = (row >= col) && (row - col <= max_attn_len);
        else
            result = abs(row - col) <= max_attn_len;

        if(min_full_attn_seqlen > 0)
            result = result || (row >= max_uih_len - min_full_attn_seqlen);

        return result;
    };

    // masking codes in device don't have to compare row/col with max_uih_len, since
    // buffer_load_xxx instruction is able to return zero for out-of-boundary access
    CK_TILE_DEVICE constexpr bool IsTokenPairInsideMask(int row, int col)
    {
        if(row < contextual_seqlen)
            return true;

        bool result = false;
        if constexpr(kUseCausal)
            result = (row >= col) && (row - col <= max_attn_len);
        else
            result = abs(row - col) <= max_attn_len;

        if(min_full_attn_seqlen > 0)
            result = result || (row >= max_uih_len - min_full_attn_seqlen);

        return result;
    };
};

template <bool kUseCausal>
struct HstuBlockMaskNoLocal
{
    static constexpr bool kUseLocal = false;
    static constexpr bool IsMasking = kUseCausal;

    int contextual_seqlen;
    int max_uih_len;

    CK_TILE_HOST_DEVICE HstuBlockMaskNoLocal(int contextual_seqlen_, int max_uih_len_)
        : contextual_seqlen(contextual_seqlen_), max_uih_len(max_uih_len_){};

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

            index_t x_end = min(i_y + YTile, max_uih_len); // for lower-triangular masking, x <= y

            return ck_tile::make_tuple(0, x_end);
        };
    }

    CK_TILE_HOST constexpr bool IsTokenPairInsideMask(int row, int col)
    {
        if(row >= max_uih_len || col >= max_uih_len)
            return false;

        if(row < contextual_seqlen)
            return true;

        if constexpr(IsMasking)
        {
            bool result = (row >= col);

            return result;
        }

        return true;
    };

    // masking codes in device don't have to compare row/col with max_uih_len, since
    // buffer_load_xxx instruction is able to return zero for out-of-boundary access
    CK_TILE_DEVICE constexpr bool IsTokenPairInsideMask(int row, int col)
    {
        if(row < contextual_seqlen)
            return true;

        if constexpr(IsMasking)
        {
            bool result = (row >= col);

            return result;
        }

        return true;
    };
};

template <bool kUseCausal, bool kUseLocal>
struct HstuBlockMasking
{
    using Type = std::conditional_t<kUseLocal,
                                    HstuBlockMaskWithLocal<kUseCausal>,
                                    HstuBlockMaskNoLocal<kUseCausal>>;
};

template <typename HstuBlockMaskType>
CK_TILE_HOST_DEVICE constexpr auto make_hstu_block_mask_with_local(int seqlen_,
                                                                   int contextual_seqlen_,
                                                                   int num_target,
                                                                   int max_attn_len_,
                                                                   int min_full_attn_seqlen_)
{
    auto max_uih_len_ = [&]() {
        if(contextual_seqlen_ > 0)
            return seqlen_ - (contextual_seqlen_ - 1) - num_target;
        else
            return seqlen_ - num_target;
    }();

    return HstuBlockMaskType{
        contextual_seqlen_, max_uih_len_, max_attn_len_, min_full_attn_seqlen_};
};

template <typename HstuBlockMaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_hstu_block_mask_without_local(int seqlen_, int contextual_seqlen_, int num_target)
{
    auto max_uih_len_ = [&]() {
        if(contextual_seqlen_ > 0)
            return seqlen_ - (contextual_seqlen_ - 1) - num_target;
        else
            return seqlen_ - num_target;
    }();

    return HstuBlockMaskType{contextual_seqlen_, max_uih_len_};
};

} // namespace ck_tile
