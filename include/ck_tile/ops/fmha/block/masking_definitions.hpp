// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "block_masking.hpp"

namespace ck_tile {

enum struct AttentionMaskEnum {
    NO_MASK = 0,

    // below enum could be causal, or sliding window
    MASK_FROM_TOP_LEFT     = 1,
    MASK_FROM_BOTTOM_RIGHT = 2,

    // this enum maybe not used by xformer/FA, since it's hard to
    // specify left/right window for varlen case. put it here for
    // debug purpose
    MASK_GENERIC,
};

template <index_t y_tile_, index_t x_tile_, bool IsLocal_>
struct DiagonalMask : MaskDefABC<DiagonalMask<y_tile_, x_tile_, IsLocal_>> {
    static constexpr index_t y_tile = y_tile_;
    static constexpr index_t x_tile = x_tile_;
    static constexpr bool IsLocal = IsLocal_;

    const index_t x;
    const index_t y;

    static constexpr const char* get_name() {
        if constexpr (IsLocal) {
            return "md";  // Name changed from mg(eneric) to md(iagonal)
        } else {
            return "mc";
        }
    }

    CK_TILE_HOST_DEVICE
    DiagonalMask(index_t x_, index_t y_)
        : x(x_), y(y_)
    {
    }

    CK_TILE_HOST_DEVICE constexpr bool
    mask_func(index_t i_y, index_t i_x) const {
        // no need to do min/max here, since i_x will never be < 0 or >= x_total
        index_t x_start = -y + i_y + 1;
        index_t x_end = i_y + x;

        if constexpr (IsLocal) {
            return i_x >= x_start && i_x < x_end;
        } else {
            return i_x < x_end;
        }
    }

    CK_TILE_HOST_DEVICE constexpr bool
    tile_mask_func(index_t i_y_tile, index_t i_x_tile) const {
        index_t top_i_y = i_y_tile * y_tile;
        index_t right_i_x = (i_x_tile + 1) * x_tile - 1;
        index_t bottom_i_y = (i_y_tile + 1) * y_tile - 1;
        index_t left_i_x = i_x_tile * x_tile;

        bool top_right_is_in = mask_func(top_i_y, right_i_x);
        bool bottom_left_is_in = mask_func(bottom_i_y, left_i_x);

        if constexpr (IsLocal) {
            // If top left isn't in then bottom right can't be in
            return top_right_is_in || bottom_left_is_in || mask_func(top_i_y, left_i_x);
        } else {
            return bottom_left_is_in;
        }
    }

    CK_TILE_HOST_DEVICE constexpr bool
    edge_tile_func(index_t i_y_tile, index_t i_x_tile) const {
        index_t top_i_y = i_y_tile * y_tile;
        index_t right_i_x = (i_x_tile + 1) * x_tile - 1;
        index_t bottom_i_y = (i_y_tile + 1) * y_tile - 1;
        index_t left_i_x = i_x_tile * x_tile;

        bool top_right_is_in = mask_func(top_i_y, right_i_x);
        bool bottom_left_is_in = mask_func(bottom_i_y, left_i_x);

        // If one corner is out and the other corner is in it's an edge tile
        if constexpr (IsLocal) {
            return top_right_is_in != bottom_left_is_in ||
                (mask_func(top_i_y, left_i_x) && (!top_right_is_in || !bottom_left_is_in));
        } else {
            return !top_right_is_in && bottom_left_is_in;
        }
    }
};

// TODO: prefer use this function in host code
// can convert from the FA style left/right to our generic coordinate
// if left_size < 0 && right_size = 0, it is normal causal mask
// local is left_size >=0 or right_size >=0
CK_TILE_HOST_DEVICE constexpr auto
make_diagonal_attention_mask_coordinates_from_lr_window(index_t left_size,
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
make_diagonal_attention_mask_from_lr_window(index_t left_size,
                                           index_t right_size,
                                           index_t y_total,
                                           index_t x_total,
                                           bool is_top_left = true)
{
    auto r = make_diagonal_attention_mask_coordinates_from_lr_window(
        left_size, right_size, y_total, x_total, is_top_left);

    typename MaskType::mask_def_t mask_def(r.at(ck_tile::number<0>{}), r.at(ck_tile::number<1>{}));
    return MaskType(mask_def, y_total, x_total);
}

} // namespace ck_tile
