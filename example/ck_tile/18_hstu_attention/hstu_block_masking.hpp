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

    // is_tile_in_first_split is false only when min_full_attn_seqlen > 0 and the current
    // tile is inside scope [max_uih_len - min_full_attn_seqlen, seqlen); for other cases
    // and tiles, is_tile_in_first_split is true
    bool is_tile_in_first_split;
    int seqlen;
    int contextual_seqlen;

    int max_attn_len;
    int min_full_attn_seqlen;

    int max_uih_len;
    int max_id;

    CK_TILE_HOST_DEVICE HstuBlockMaskWithLocal(bool is_tile_in_first_split_,
                                               int seqlen_,
                                               int contextual_seqlen_,
                                               int max_attn_len_,
                                               int min_full_attn_seqlen_,
                                               int num_target_)
        : is_tile_in_first_split(is_tile_in_first_split_),
          seqlen(seqlen_),
          contextual_seqlen(contextual_seqlen_),
          max_attn_len(max_attn_len_),
          min_full_attn_seqlen(min_full_attn_seqlen_)
    {
        max_uih_len = seqlen - num_target_;

        if(contextual_seqlen > 0)
            max_id = max_uih_len - (contextual_seqlen - 1);
        else
            max_id = max_uih_len;
    };

    // to get the loop length along X axis, return index:[start, end), end-start=length
    // use this if need loop over X axis tile by tile (eg. seqlen_k loop-over)
    // i_y is the start offset of the current tile along the seqlen_q dimension
    template <index_t YTile, index_t XTile>
    CK_TILE_HOST_DEVICE constexpr auto
    GetTileRangeAlongX(index_t i_y, number<YTile>, number<XTile>) const
    {
        if constexpr(kUseCausal)
        {
            if(!is_tile_in_first_split)
            {
                index_t x_end = min(i_y + YTile, seqlen);
                return ck_tile::make_tuple(0, x_end);
            };
        };

        if constexpr(!kUseCausal)
        {
            if(i_y >= contextual_seqlen)
            {
                if(i_y < max_uih_len)
                {
                    index_t x_start         = max(0, i_y - max_attn_len);
                    index_t x_start_aligned = x_start - x_start % XTile;

                    if(i_y + YTile > max_uih_len - max_attn_len)
                    {
                        return ck_tile::make_tuple(x_start_aligned, seqlen);
                    }
                    else
                    {
                        index_t x_end = min(i_y + YTile + max_attn_len, seqlen);
                        return ck_tile::make_tuple(x_start_aligned, x_end);
                    };
                }
                else
                {
                    index_t x_start = i_y - max_attn_len;
                    index_t x_end   = seqlen;

                    return ck_tile::make_tuple(x_start - x_start % XTile, x_end);
                }
            }
            else
            {
                if(i_y + YTile > max_uih_len)
                {
                    index_t x_end = min(i_y + YTile, seqlen);
                    return ck_tile::make_tuple(0, x_end);
                }
                else
                {
                    index_t x_end = max(i_y + YTile + max_attn_len, max_uih_len);
                    return ck_tile::make_tuple(0, x_end);
                };
            }
        }
        else // kUseCausal && kUseLocal
        {
            if(i_y >= contextual_seqlen)
            {
                index_t x_end = min(i_y + YTile, seqlen);

                if(i_y < max_uih_len)
                {
                    index_t x_start = max(0, i_y - max_attn_len);
                    return ck_tile::make_tuple(x_start - x_start % XTile, x_end);
                }
                else
                {
                    index_t x_start = max_uih_len - max_attn_len;
                    return ck_tile::make_tuple(x_start - x_start % XTile, x_end);
                }
            }
            else
            {
                index_t x_end = min(i_y + YTile, seqlen);

                if(i_y + YTile > max_uih_len)
                {
                    return ck_tile::make_tuple(0, x_end);
                }
                else
                {
                    return ck_tile::make_tuple(0, max_uih_len);
                };
            }
        };
    }

    CK_TILE_HOST bool IsTokenPairInsideMask(int row, int col)
    {
        int row_id;
        int col_id;

        if(contextual_seqlen > 0)
        {
            // row_id/col_id is clamped from physical row/col according to contextual_seqlen and
            // max_uih_len
            row_id = max(row - contextual_seqlen + 1, 0);
            col_id = max(col - contextual_seqlen + 1, 0);

            row_id = min(row_id, max_id);
            col_id = min(col_id, max_id);

            if(row_id == 0 && col_id < max_id)
                return true;
        }
        else
        {
            // row_id/col_id is clamped from physical row/col according to contextual_seqlen and
            // max_uih_len
            row_id = min(row, max_id);
            col_id = min(col, max_id);
        };

        // use row_id/col_id to check the dist between two q/k token pair, token pairs on the
        // diagonal line are always considerred
        if constexpr(kUseCausal)
        {
            bool in_min_full_scope =
                (min_full_attn_seqlen > 0) ? (row_id >= max_id - min_full_attn_seqlen) : false;

            return (((row_id > col_id) || (row == col)) &&
                    ((row_id - col_id <= max_attn_len) || in_min_full_scope));
        }
        else
        {
            bool in_min_full_scope =
                (min_full_attn_seqlen > 0) ? (row_id >= max_id - min_full_attn_seqlen) : false;

            return (((row_id != col_id) || (row == col)) &&
                    ((abs(row_id - col_id) <= max_attn_len) || in_min_full_scope));
        }
    };

    CK_TILE_DEVICE int IsTokenPairInsideMask(int row, int col)
    {
        int row_id;
        int col_id;

        if(contextual_seqlen > 0)
        {
            // row_id/col_id is clamped from physical row/col according to contextual_seqlen and
            // max_uih_len
            row_id = max(row - contextual_seqlen + 1, 0);
            col_id = max(col - contextual_seqlen + 1, 0);

            row_id = min(row_id, max_id);
            col_id = min(col_id, max_id);

            if(row_id == 0 && col_id < max_id)
                return 1;
        }
        else
        {
            // row_id/col_id is clamped from physical row/col according to contextual_seqlen and
            // max_uih_len
            row_id = min(row, max_id);
            col_id = min(col, max_id);
        };

        // use row_id/col_id to check the dist between two q/k token pair, token pairs on the
        // diagonal line are always considerred
        if constexpr(kUseCausal)
        {
            bool in_min_full_scope = !is_tile_in_first_split;

            bool res = (((row_id > col_id) || (row == col)) &&
                        ((row_id - col_id <= max_attn_len) || in_min_full_scope));

            return static_cast<int>(res);
        }
        else
        {
            bool in_min_full_scope = !is_tile_in_first_split;

            bool res = (((row_id != col_id) || (row == col)) &&
                        ((abs(row_id - col_id) <= max_attn_len) || in_min_full_scope));

            return static_cast<int>(res);
        }
    };

    // if the whole tile inside the masking area, no need for pixel-by-pixel checking
    template <index_t TileWidth, index_t TileHeight>
    CK_TILE_DEVICE constexpr bool IsFullTileInsideMask(index_t i_tile_top,
                                                       index_t i_tile_left,
                                                       number<TileWidth>,
                                                       number<TileHeight>) const
    {
        if constexpr(kUseCausal)
        {
            index_t i_tile_right = i_tile_left + TileWidth;

            if(!is_tile_in_first_split && i_tile_right <= min(i_tile_top + 1, max_uih_len))
                return true;
        }
        else
        {
            index_t i_tile_right  = i_tile_left + TileWidth;
            index_t i_tile_bottom = i_tile_top + TileHeight;

            if(!is_tile_in_first_split && i_tile_bottom <= max_uih_len &&
               i_tile_right <= i_tile_top + 1)
                return true;
        };

        return false;
    }
};

template <bool kUseCausal>
struct HstuBlockMaskNoLocal
{
    static constexpr bool kUseLocal = false;
    static constexpr bool IsMasking = kUseCausal;

    int seqlen;
    int contextual_seqlen;

    int max_uih_len;
    int max_id;

    CK_TILE_HOST_DEVICE HstuBlockMaskNoLocal(int seqlen_, int contextual_seqlen_, int num_target_)
        : seqlen(seqlen_), contextual_seqlen(contextual_seqlen_)
    {
        max_uih_len = seqlen - num_target_;

        if(contextual_seqlen > 0)
            max_id = max_uih_len - (contextual_seqlen - 1);
        else
            max_id = max_uih_len;
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
            return ck_tile::make_tuple(0, seqlen);
        }
        else
        {
            index_t x_end = min(i_y + YTile, seqlen);

            if(i_y < contextual_seqlen)
            {
                if(i_y + YTile > max_uih_len)
                {
                    return ck_tile::make_tuple(0, x_end);
                }
                else
                {
                    return ck_tile::make_tuple(0, max_uih_len);
                };
            }
            else
            {
                return ck_tile::make_tuple(0, x_end);
            };
        };
    }

    CK_TILE_HOST bool IsTokenPairInsideMask(int row, int col)
    {
        int row_id;
        int col_id;

        if(contextual_seqlen > 0)
        {
            // row_id/col_id is clamped from physical row/col according to contextual_seqlen and
            // max_uih_len
            row_id = max(row - contextual_seqlen + 1, 0);
            col_id = max(col - contextual_seqlen + 1, 0);

            row_id = min(row_id, max_id);
            col_id = min(col_id, max_id);

            if(row_id == 0 && col_id < max_id)
                return true;
        }
        else
        {
            // row_id/col_id is clamped from physical row/col according to contextual_seqlen and
            // max_uih_len
            row_id = min(row, max_id);
            col_id = min(col, max_id);
        };

        // use row_id/col_id to check the dist between two q/k token pair, token pairs on the
        // diagonal line are always considerred
        if constexpr(IsMasking)
        {
            return (row_id > col_id) || (row == col);
        }
        else
        {
            return (row_id != col_id) || (row == col);
        };
    };

    CK_TILE_DEVICE int IsTokenPairInsideMask(int row, int col)
    {
        int row_id;
        int col_id;

        if(contextual_seqlen > 0)
        {
            // row_id/col_id is clamped from physical row/col according to contextual_seqlen and
            // max_uih_len
            row_id = max(row - contextual_seqlen + 1, 0);
            col_id = max(col - contextual_seqlen + 1, 0);

            row_id = min(row_id, max_id);
            col_id = min(col_id, max_id);

            if(row_id == 0 && col_id < max_id)
                return 1;
        }
        else
        {
            // row_id/col_id is clamped from physical row/col according to contextual_seqlen and
            // max_uih_len
            row_id = min(row, max_id);
            col_id = min(col, max_id);
        };

        // use row_id/col_id to check the dist between two q/k token pair, token pairs on the
        // diagonal line are always considerred
        if constexpr(IsMasking)
        {
            bool res = ((row_id > col_id) || (row == col));

            return static_cast<int>(res);
        }
        else
        {
            bool res = ((row_id != col_id) || (row == col));

            return static_cast<int>(res);
        };
    };

    // if the whole tile inside the masking area, no need for pixel-by-pixel checking
    template <index_t TileWidth, index_t TileHeight>
    CK_TILE_DEVICE constexpr bool IsFullTileInsideMask(index_t i_tile_top,
                                                       index_t i_tile_left,
                                                       number<TileWidth>,
                                                       number<TileHeight>) const
    {
        if constexpr(kUseCausal)
        {
            index_t i_tile_right  = i_tile_left + (TileWidth - 1);
            index_t i_tile_bottom = i_tile_top + (TileHeight - 1);

            // assume num_target > 0 with high probability, don't check whether num_target is 0;
            // so if num_target is 0, IsTokenPairInsideMask() will be called for the bottom tile
            if(i_tile_bottom >= max_uih_len || i_tile_right > i_tile_top)
                return false;

            return true;
        }
        else
        {
            index_t i_tile_right  = i_tile_left + (TileWidth - 1);
            index_t i_tile_bottom = i_tile_top + (TileHeight - 1);

            // assume num_target > 0 with high probability, don't check whether num_target is 0;
            // so if num_target is 0, IsTokenPairInsideMask() will be called for the bottom tile
            if(i_tile_bottom >= max_uih_len || i_tile_right >= max_uih_len)
                return false;

            return true;
        }
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
CK_TILE_HOST_DEVICE constexpr auto make_hstu_block_mask_with_local(bool is_tile_in_first_split_,
                                                                   int seqlen_,
                                                                   int contextual_seqlen_,
                                                                   int num_target,
                                                                   int max_attn_len_,
                                                                   int min_full_attn_seqlen_)
{
    return HstuBlockMaskType{is_tile_in_first_split_,
                             seqlen_,
                             contextual_seqlen_,
                             max_attn_len_,
                             min_full_attn_seqlen_,
                             num_target};
};

template <typename HstuBlockMaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_hstu_block_mask_without_local(int seqlen_, int contextual_seqlen_, int num_target)
{
    return HstuBlockMaskType{seqlen_, contextual_seqlen_, num_target};
};

} // namespace ck_tile
