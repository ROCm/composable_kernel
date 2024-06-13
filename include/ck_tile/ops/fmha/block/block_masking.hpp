// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// FIXME: `end - start` doesn't actually reflect how many nonzeroes are in the row mask
template <typename func_t>
struct IndexIterator {
    CK_TILE_HOST_DEVICE IndexIterator(index_t start_, index_t end_, func_t predicate_)
        : start(start_), end(end_), current(start_), predicate(predicate_)
    {
        if (!predicate(current)) { advance(); }
    }

    CK_TILE_HOST_DEVICE constexpr bool
    at_end() const { return !(current < end); }

    CK_TILE_HOST_DEVICE constexpr void
    advance() {
        if (!at_end()) {
            do {
                current++;
            } while (!predicate(current) && !at_end());
        }
    }

    const index_t start;
    const index_t end;
    index_t current;

    private:
    func_t predicate;
};

// This struct isn't necessary, but it lays out the requirements of a MaskDef
template <typename derived_t>
struct MaskDefABC {
    CK_TILE_HOST_DEVICE constexpr bool
    mask_func(index_t i_y, index_t i_x) const {
        return static_cast<derived_t*>(this)->mask_func(i_y, i_x);
    }

    CK_TILE_HOST_DEVICE constexpr bool
    tile_mask_func(index_t i_y_tile, index_t i_x_tile) const {
        return static_cast<derived_t*>(this)->tile_mask_func(i_y_tile, i_x_tile);
    }

    CK_TILE_HOST_DEVICE constexpr bool
    edge_tile_func(index_t i_y_tile, index_t i_x_tile) const {
        return static_cast<derived_t*>(this)->edge_tile_func(i_y_tile, i_x_tile);
    }
};


// It's a lot harder to do names for all specialized templates now, so I define them in the structs instead
// namespace impl {
//     // Set name for no masking, otherwise use mask def name
//     template <typename mask_def_t, bool IsMasking_> struct MaskName;
//     template<typename mask_def_t> struct MaskName<mask_def_t, false> { static constexpr const char * name = "mn"; };
//     template<typename mask_def_t> struct MaskName<mask_def_t, true> { static constexpr const char * name = mask_def_t::name; };
// }

template <typename mask_def_t_,
          index_t y_tile_,
          index_t x_tile_,
          bool IsTileMask_,
          bool IsMasking_ = true>
struct GenericAttentionMask
{
    using mask_def_t = mask_def_t_;

    static constexpr index_t y_tile = y_tile_;
    static constexpr index_t x_tile = x_tile_;

    static constexpr bool IsMasking = IsMasking_;
    static constexpr bool IsTileMask = IsTileMask_;

    const index_t y_total, x_total;
    const index_t y_tile_total, x_tile_total;

    static constexpr const char* get_name() {
        if constexpr (IsMasking) {
            return mask_def_t::get_name();
        } else {
            return "mn";
        }
    }
    static constexpr const char* name = get_name();

    CK_TILE_HOST_DEVICE
    GenericAttentionMask(mask_def_t mask_def_,
                         index_t y_total_,
                         index_t x_total_) :
        y_total(y_total_),
        x_total(x_total_),
        y_tile_total(integer_divide_ceil(y_total_, y_tile)),
        x_tile_total(integer_divide_ceil(x_total_, x_tile)),
        mask_def(mask_def_)
    {
        static_assert(std::is_base_of<MaskDefABC<mask_def_t>, mask_def_t>::value,
                      "mask definition is not derived from MaskDefABC");
    }

    CK_TILE_HOST_DEVICE constexpr auto
    ElementwiseMask(index_t i_y, index_t i_x) const
    {
        static_assert(!IsTileMask, "tile mask shouldn't be evaluated elementwise");

        bool in_bounds = i_y < y_total &&
                         i_x < x_total &&
                         i_y >= 0 &&
                         i_x >= 0;

        if constexpr (!IsMasking) {
            return in_bounds;
        } else if constexpr (IsTileMask) {
            return in_bounds && TileMask(
                integer_divide_floor(i_y, y_tile),  // Find which tile the index belongs to
                integer_divide_floor(i_x, x_tile)
            );
        } else {
            return in_bounds && mask_def.mask_func(i_y, i_x);
        }
    }

    CK_TILE_HOST_DEVICE constexpr auto
    TileMask(index_t i_y_tile, index_t i_x_tile) const
    {
        bool in_bounds = i_y_tile < y_tile_total &&
                         i_x_tile < x_tile_total &&
                         i_y_tile >= 0 &&
                         i_x_tile >= 0;

        if constexpr (!IsMasking) {
            return in_bounds;
        } else {
            return in_bounds && mask_def.tile_mask_func(i_y_tile, i_x_tile);
        }
    }

    CK_TILE_HOST_DEVICE constexpr auto
    GetTileIndexIteratorAlongX(index_t i_y_tile) const
    {
        auto tile_row_pred = [i_y_tile, this](index_t i_x_tile){ return TileMask(i_y_tile, i_x_tile); };
        return IndexIterator<decltype(tile_row_pred)>(0, x_tile_total, tile_row_pred);
    }

    CK_TILE_HOST_DEVICE constexpr auto
    IsEdgeTile(index_t i_y_tile, index_t i_x_tile) const
    {
        if constexpr (!IsMasking || IsTileMask) {  // No edge tiles in tile mask
            return false;
        } else {
            bool in_bounds = i_y_tile < y_tile_total &&
                             i_x_tile < x_tile_total &&
                             i_y_tile >= 0 &&
                             i_x_tile >= 0;

            return in_bounds && mask_def.edge_tile_func(i_y_tile, i_x_tile);
        }
    }

    private:
    const mask_def_t mask_def;
};

} // namespace ck_tile
