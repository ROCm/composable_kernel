// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

enum struct GenericAttentionMaskEnum
{
    NO_MASK = 0,

    // below enum could be causal, or sliding window
    MASK_FROM_TOP_LEFT     = 1,
    MASK_FROM_BOTTOM_RIGHT = 2,

    // this enum maybe not used by xformer/FA, since it's hard to
    // specify left/right window for varlen case. put it here for
    // debug purpose
    MASK_GENERIC,
};

// clang-format off
/*  generic Attention Mask Coordinate
    use x(horizontal axis), y(vertical axis) to describe mask.
    top-left corner is origin

    x=1/y=5(top-left)  x=4/y=5(botm-r)    x=6/y=5            x=8/y=5(no mask)
    1 * * * * * * *    1 1 1 1 * * * *    1 1 1 1 1 1 * *    1 1 1 1 1 1 1 1
    1 1 * * * * * *    1 1 1 1 1 * * *    1 1 1 1 1 1 1 *    1 1 1 1 1 1 1 1
    1 1 1 * * * * *    1 1 1 1 1 1 * *    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1
    1 1 1 1 * * * *    1 1 1 1 1 1 1 *    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1
    1 1 1 1 1 * * *    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1
    l=7,-1/r=0(tl)     l=7,-1/r=0(br)

    x=1/y=2            x=4/y=2            x=6/y=2            x=8/y=2
    1 * * * * * * *    1 1 1 1 * * * *    1 1 1 1 1 1 * *    1 1 1 1 1 1 1 1
    1 1 * * * * * *    1 1 1 1 1 * * *    1 1 1 1 1 1 1 *    1 1 1 1 1 1 1 1
    * 1 1 * * * * *    * 1 1 1 1 1 * *    * 1 1 1 1 1 1 1    * 1 1 1 1 1 1 1
    * * 1 1 * * * *    * * 1 1 1 1 1 *    * * 1 1 1 1 1 1    * * 1 1 1 1 1 1
    * * * 1 1 * * *    * * * 1 1 1 1 1    * * * 1 1 1 1 1    * * * 1 1 1 1 1
    l=1/r=0(tl)        l=1/r=3(tl)        l=1/r=5(tl)        l=1/r=7(tl)
                       l=4/r=0(br)        l=4/r=2(br)        l=4/r=4(br)

                       x=4/y=-1           x=6/y=-1            x=8/y=-1
                       * * 1 1 * * * *    * * 1 1 1 1 * *    * * 1 1 1 1 1 1
                       * * * 1 1 * * *    * * * 1 1 1 1 *    * * * 1 1 1 1 1
                       * * * * 1 1 * *    * * * * 1 1 1 1    * * * * 1 1 1 1
                       * * * * * 1 1 *    * * * * * 1 1 1    * * * * * 1 1 1
                       * * * * * * 1 1    * * * * * * 1 1    * * * * * * 1 1

    x=-2/y=5           x=1/y=5(top-left)  x=0/y=5(botm-r)
    * * * * * * * *    1 * * *            * * * *
    * * * * * * * *    1 1 * *            1 * * *
    * * * * * * * *    1 1 1 *            1 1 * *
    1 * * * * * * *    1 1 1 1            1 1 1 *
    1 1 * * * * * *    1 1 1 1            1 1 1 1

    Validations:
        x + y > 1 (x + y >= 2)

    Note:
        y = seq_q, x = 1 -> top-left
        y = seq_q, x = seq_k - seq_q + 1 -> bottom-right
        y < seq_q, x < seq_k -> local-attn
        y = seq_q, x = seq_k -> no mask

*/
namespace impl {
    template <bool IsMasking_, bool IsLocal_> struct MaskName;
    template<> struct MaskName<false, false> { static constexpr const char * name = "mn"; };
    template<> struct MaskName<false, true> { static constexpr const char * name = "mn"; };
    template<> struct MaskName<true, false> { static constexpr const char * name = "mc"; };
    template<> struct MaskName<true, true> { static constexpr const char * name = "mg"; };
}
// clang-format on

// FIXME: Length doesn't actually reflect how many nonzeroes are in this row mask
template <typename func_t>
struct IndexIterator {
    CK_TILE_HOST_DEVICE IndexIterator(index_t start_, index_t end_, func_t predicate_)
        : start(start_), end(end_), current(start_), length(end_ - start_), predicate(predicate_)
    {
        if(!predicate(current)) { advance(); }
    }

    CK_TILE_HOST_DEVICE bool at_end() const { return !(current < end); }

    CK_TILE_HOST_DEVICE void advance() {
        if (!at_end()) {
            do {
                current++;
            } while (!predicate(current) && !at_end());
        }
    }

    const index_t start;
    const index_t end;
    index_t current;
    const index_t length;

    private:
    func_t predicate;
};

template <bool IsMasking_ = true, bool IsLocal_ = false>
struct GenericAttentionMask
{
    static constexpr bool IsMasking = IsMasking_; // false will disable masking
    static constexpr bool IsLocal   = IsLocal_;   // if true, upper/lower area could have mask,
                                                  // else only upper-right could have mask

    static constexpr const char* name = impl::MaskName<IsMasking, IsLocal>::name;

    // TODO: What't the best way to make these "static constexpr"?
    const index_t y_tile;
    const index_t x_tile;

    CK_TILE_HOST_DEVICE GenericAttentionMask(index_t y_total_, index_t x_total_, index_t y_tile_, index_t x_tile_)
        : GenericAttentionMask(0, 0, y_total_, x_total_, y_tile_, x_tile_)
    {
    }

    CK_TILE_HOST_DEVICE
    GenericAttentionMask(index_t y_, index_t x_, index_t y_total_, index_t x_total_, index_t y_tile_, index_t x_tile_)
        : y(y_), x(x_), y_total(y_total_), x_total(x_total_), y_tile(y_tile_), x_tile(x_tile_)
    {
    }
    template <typename MaskCoordinates>
    CK_TILE_HOST_DEVICE GenericAttentionMask(const MaskCoordinates& mask_coord, index_t y_tile_, index_t x_tile_)
        : y(mask_coord.at(number<0>{})),
          x(mask_coord.at(number<1>{})),
          y_total(mask_coord.at(number<2>{})),
          x_total(mask_coord.at(number<3>{})),
          y_tile(y_tile_), x_tile(x_tile_)
    {
    }

    // to get the loop length along X axis, return index:[start, end), end-start=length
    // use this if need loop over X axis tile by tile (like k-seqlen loopover)
    // TODO: x_end still could be negative, so end-start could be negative(need check)
    CK_TILE_HOST_DEVICE constexpr auto
    GetTileIndexIteratorAlongX(index_t i_y) const
    {
        if constexpr(!IsMasking)
        {
            index_t x_tiles = integer_divide_floor(x_total, x_tile);

            auto full_pred = [x_tiles](index_t i_x_tile) {
                if (i_x_tile >= 0 && i_x_tile < x_tiles) { return true; }
                return false;
            };

            return IndexIterator<decltype(full_pred)>(0, x_tiles, full_pred);
        }
        else
        {
            // get the tile start/end range assum we loop over along X tile by tile
            index_t x_start = [&]() {
                index_t tmp = max(-y + i_y + 1, 0);
                return (tmp / x_tile) * x_tile; // round to tile aligned
            }();

            // TODO: end could be negative, we ignore clamp here, and let caller to check
            //      ... in which case end-start is negative
            index_t x_end = [&]() {
                index_t tmp = min(i_y + y_tile - 1 + x, x_total);
                return ((tmp + x_tile - 1) / x_tile) * x_tile;
            }();

            auto contiguous_pred = [x_start, x_end](index_t i_x_tile) {
                if (i_x_tile >= x_start && i_x_tile < x_end) { return true; }
                return false;
            };

            return IndexIterator<decltype(contiguous_pred)>(x_start, x_end, contiguous_pred);
        }
    }

    // per-pixel check if out-of-bound, if true, need mask a value(like -INF)
    CK_TILE_HOST_DEVICE constexpr auto IsOutOfBound(index_t i_y, index_t i_x) const
    {
        if constexpr(!IsMasking)
        {
            return i_x >= x_total;
        }
        else
        {
            // no need to do min/max here, since i_x will never be < 0 or >= x_total
            index_t x_start = -y + i_y + 1;
            index_t x_end   = min(i_y + x, x_total);

            if constexpr(IsLocal)
            {
                return i_x < x_start || i_x >= x_end;
            }
            else
            {
                return i_x >= x_end;
            }
        }
    }

    // if current tile is at the edge, means need per-pixel mask check.
    // otherwise no need to check per-pixel
    // Attention! assume the idex passed in this function is with in range of GetTileRangeAlongX()
    // can be used as a fast-path to decide if do per-pixel check or not
    template <index_t TileHeight, index_t TileWidth>
    CK_TILE_HOST_DEVICE constexpr auto
    IsEdgeTile(index_t i_tile_top, index_t i_tile_left, number<TileHeight>, number<TileWidth>) const
    {
        if constexpr(IsLocal)
        {
            // check top-right corner > x or left-borrom corner < x
            index_t i_tile_right  = i_tile_left + TileWidth;
            index_t i_tile_bottom = i_tile_top + TileHeight;
            index_t x_end         = min(i_tile_top + x, x_total);

            bool top_right_edge          = i_tile_right > (i_tile_top + x);
            bool bottom_left_edge        = i_tile_bottom > (i_tile_left + y);
            bool is_partial_out_of_bound = i_tile_right > x_end; // only consider right-pad for now

            return top_right_edge || bottom_left_edge || is_partial_out_of_bound;
        }
        else
        {
            // only need to check top-right corner > x
            index_t i_tile_right = i_tile_left + TileWidth;
            index_t x_end        = min(i_tile_top + x, x_total);

            bool top_right_edge = i_tile_right > x_end;
            return top_right_edge;
        }
    }

    private:
    index_t y, x;
    index_t y_total, x_total;
};

// clang-format off
namespace impl {
    template <bool IsMasking_> struct SimplifiedMaskName;
    template<> struct SimplifiedMaskName<false> { static constexpr const char * name = "nomask"; };
    template<> struct SimplifiedMaskName<true> { static constexpr const char * name = "mask"; };
}

template <bool IsMasking_ = true>
struct SimplifiedGenericAttentionMask
{
    static constexpr bool IsMasking = IsMasking_; // false will disable masking

    static constexpr const char* name = impl::SimplifiedMaskName<IsMasking>::name;

    // TODO: What't the best way to make these "static constexpr"?
    const index_t y_tile;
    const index_t x_tile;

    CK_TILE_HOST_DEVICE SimplifiedGenericAttentionMask(index_t y_total_, index_t x_total_, index_t y_tile_, index_t x_tile_)
        : SimplifiedGenericAttentionMask(0, 0, y_total_, x_total_, y_tile_, x_tile_)
    {
    }

    CK_TILE_HOST_DEVICE
    SimplifiedGenericAttentionMask(index_t y_, index_t x_, index_t y_total_, index_t x_total_, index_t y_tile_, index_t x_tile_)
        : y(y_), x(x_), y_total(y_total_), x_total(x_total_), y_tile(y_tile_), x_tile(x_tile_)
    {
    }
    template <typename MaskCoordinates>
    CK_TILE_HOST_DEVICE SimplifiedGenericAttentionMask(const MaskCoordinates& mask_coord, index_t y_tile_, index_t x_tile_)
        : y(mask_coord.at(number<0>{})),
          x(mask_coord.at(number<1>{})),
          y_total(mask_coord.at(number<2>{})),
          x_total(mask_coord.at(number<3>{})),
          y_tile(y_tile_), x_tile(x_tile_)
    {
    }

    CK_TILE_HOST_DEVICE constexpr auto
    GetTileIndexIteratorAlongX(index_t i_y) const
    {
        if constexpr(!IsMasking)
        {
            index_t x_tiles = integer_divide_floor(x_total, x_tile);

            auto full_pred = [x_tiles](index_t i_x_tile) {
                if (i_x_tile >= 0 && i_x_tile < x_tiles) { return true; }
                return false;
            };

            return IndexIterator<decltype(full_pred)>(0, x_tiles, full_pred);
        }
        else
        {
            // get the tile start/end range assum we loop over along X tile by tile
            index_t x_start = [&]() {
                index_t tmp = max(-y + i_y + 1, 0);
                return (tmp / x_tile) * x_tile; // round to tile aligned
            }();

            // TODO: end could be negative, we ignore clamp here, and let caller to check
            //      ... in which case end-start is negative
            index_t x_end = [&]() {
                index_t tmp = min(i_y + y_tile - 1 + x, x_total);
                return ((tmp + x_tile - 1) / x_tile) * x_tile;
            }();

            auto contiguous_pred = [x_start, x_end](index_t i_x_tile) {
                if (i_x_tile >= x_start && i_x_tile < x_end) { return true; }
                return false;
            };

            return IndexIterator<decltype(contiguous_pred)>(x_start, x_end, contiguous_pred);
        }
    }

    // per-pixel check if out-of-bound, if true, need mask a value(like -INF)
    CK_TILE_HOST_DEVICE constexpr auto IsOutOfBound(index_t i_y, index_t i_x) const
    {
        if constexpr(!IsMasking)
        {
            // the only case that need do following compare is under kPadSeqLenK
            // ... for non-masking kernel.
            return i_x >= x_total;
        }
        else
        {
            index_t x_start = -y + i_y + 1;          // this could be negative, but it's fine
            index_t x_end   = min(i_y + x, x_total); // need min in case x is padded

            return i_x < x_start || i_x >= x_end;
        }
    }

    // if current tile is at the edge, means need per-pixel mask check.
    // otherwise no need to check per-pixel
    // Attention! assume the idex passed in this function is with in range of GetTileRangeAlongX()
    // can be used as a fast-path to decide if do per-pixel check or not
    CK_TILE_HOST_DEVICE constexpr auto
    IsEdgeTile(index_t i_y, index_t i_x) const
    {
        if constexpr(!IsMasking)
        {
            // the only case that need do following compare is under kPadSeqLenK
            // ... for non-masking kernel.
            // return (i_x < x_total) && ((i_x + x_tile) > x_total);

            // TODO: no need to check begin
            return (i_x + x_tile) > x_total;
        }
        else
        {
            // check top-right corner > x or left-borrom corner < x
            index_t i_x_end = i_x + x_tile;
            index_t i_y_end = i_y + y_tile;
            // index_t x_end    = min(i_y + x, x_total);

            bool top_right_edge   = i_x_end > min(i_y + x, x_total); // consider right pad
            bool bottom_left_edge = i_y_end > (i_x + y);
            // bool is_partial_out_of_bound = i_x_end > x_end; // only consider right-pad for now

            return top_right_edge || bottom_left_edge;
        }
    }

    private:
    const index_t y, x;
    const index_t y_total, x_total;
};

// template <bool IsMasking_ = true>
// struct BigBirdAttentionMask {
//     static constexpr bool IsMasking = IsMasking_; // false will disable masking

//     static constexpr const char* name = impl::SimplifiedMaskName<IsMasking>::name;

//     // TODO: What't the best way to make these "static constexpr"?
//     const index_t y_tile;
//     const index_t x_tile;

//     CK_TILE_HOST_DEVICE
//     SimplifiedGenericAttentionMask(index_t y_total_, index_t x_total_, index_t y_tile_, index_t x_tile_)
//         : y_total(y_total_), x_total(x_total_), y_tile(y_tile_), x_tile(x_tile_)
//     {
//     }

//     CK_TILE_HOST_DEVICE constexpr auto
//     GetTileIndexIteratorAlongX(index_t i_y) const
//     {
//         if constexpr(!IsMasking)
//         {
//             index_t x_tiles = integer_divide_floor(x_total, x_tile);

//             auto full_pred = [x_tiles](index_t i_x_tile) {
//                 if (i_x_tile >= 0 && i_x_tile < x_tiles) { return true; }
//                 return false;
//             };

//             return IndexIterator<decltype(full_pred)>(0, x_tiles, full_pred);
//         }
//         else
//         {
//             index_t g = 2;
//             index_t w = 3;
//             index_t r = 3;  // FIXME: Not using randomly generated values currently
//             index_t y_block;
//             index_t x_block;

//             index_t i_y_tile = i_y / y_tile;

//             auto big_bird_pred = [](indedx_t i_x_tile) {
//                 if (i_x_tile < g || i_y_tile < g) { return true; }
//                 if (ck::math::abs(i_x_tile - i_y_tile) < ck::math::integer_divide_ceil(w, 2)) { return true; }
//                 if (i_y_tile == 6 || i_y_tile == 31 || i_y_tile == 43) { return true; }  // FIXME: Use randomly generated values rather than hardcoded ones
//                 return false;
//             };
//             return IndexIterator<decltype(contiguous_pred)>(x_start, x_end, contiguous_pred);
//         }
//     }

//     // per-pixel check if out-of-bound, if true, need mask a value(like -INF)
//     CK_TILE_HOST_DEVICE constexpr auto IsOutOfBound(index_t i_y, index_t i_x) const
//     {
//         if constexpr(!IsMasking)
//         {
//         }
//         else
//         {
//         }
//     }

//     // if current tile is at the edge, means need per-pixel mask check.
//     // otherwise no need to check per-pixel
//     // Attention! assume the idex passed in this function is with in range of GetTileRangeAlongX()
//     // can be used as a fast-path to decide if do per-pixel check or not
//     CK_TILE_HOST_DEVICE constexpr auto
//     IsEdgeTile(index_t i_y, index_t i_x) const
//     {
//         return false;  // FIXME: For now we are going to make the mask blocks line up with the tiles, so there should be no edge tiles.
//     }

//     private:
//     const index_t y_total, x_total;
// };

// TODO: prefer use this function in host code
// can convert from the FA style left/right to our generic coordinate
// if left_size < 0 && right_size = 0, it is normal causal mask
// local is left_size >=0 or right_size >=0
CK_TILE_HOST_DEVICE constexpr auto
make_generic_attention_mask_coordinates_from_lr_window(index_t left_size,
                                                       index_t right_size,
                                                       index_t y_total,
                                                       index_t x_total,
                                                       bool is_top_left = true)
{
    // TODO: below should all use sgpr arithmetic
    index_t left_size_tmp  = is_top_left ? y_total - 1 : x_total - 1;
    index_t right_size_tmp = is_top_left ? x_total - 1 : y_total - 1;

    left_size  = left_size < 0 ? left_size_tmp : left_size;
    right_size = right_size < 0 ? right_size_tmp : right_size;

    index_t x_tmp = is_top_left ? 0 : x_total - y_total;
    index_t y_tmp = is_top_left ? 0 : y_total - x_total;

    index_t x = 1 + right_size + x_tmp;
    index_t y = 1 + left_size + y_tmp;

    return ck_tile::make_tuple(y, x, y_total, x_total);
}

template <typename MaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_generic_attention_mask_from_lr_window(index_t left_size,
                                           index_t right_size,
                                           index_t y_total,
                                           index_t x_total,
                                           index_t y_mask,
                                           index_t x_mask,
                                           bool is_top_left = true)
{
    auto r = make_generic_attention_mask_coordinates_from_lr_window(
        left_size, right_size, y_total, x_total, is_top_left);
    return MaskType{r.at(ck_tile::number<0>{}), r.at(ck_tile::number<1>{}), y_total, x_total, y_mask, x_mask};
}
} // namespace ck_tile
