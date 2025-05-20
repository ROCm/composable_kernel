// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/utility.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

/**
 * @brief This class provides description of tile windowed view on the device memory.
 *
 * @note This class does not provide any functions to read or modify device memory.
 *
 * @tparam BottomTensorView_    Class describing & holding device tensor memory.
 * @tparam WindowLengths_       Spatial sizes of windowed view on tensor.
 */
template <typename TileWindowType_, typename BottomTensorView_, typename WindowLengths_>
struct tile_window_base
{
    // window_origin
    using BottomTensorView                    = remove_reference_t<BottomTensorView_>;
    using BottomTensorDesc                    = typename BottomTensorView::TensorDesc;
    using WindowLengths                       = remove_cvref_t<WindowLengths_>;
    static constexpr index_t NDimBottomTensor = BottomTensorDesc::get_num_of_dimension();
    using BottomTensorIndex                   = array<index_t, NDimBottomTensor>;

    using DataType = remove_cvref_t<typename BottomTensorView::DataType>;

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;

    // window_lengths

    static_assert(ck_tile::is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    WindowLengths window_lengths_;

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    CK_TILE_DEVICE constexpr auto get_window_origin() const { return window_origin_; }
    CK_TILE_DEVICE constexpr auto get_window_lengths() const { return window_lengths_; }
    CK_TILE_DEVICE constexpr auto get_bottom_tensor_view() const { return bottom_tensor_view_; }
    CK_TILE_DEVICE static constexpr index_t get_num_of_dimension() { return NDimBottomTensor; }

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& new_window_origin)
    {
        window_origin_ = new_window_origin;

        // Delegate to child if it implements extra logic
        static_cast<TileWindowType_*>(this)->set_window_origin_extended(new_window_origin);
    }
    // Default no-op; can be overridden in child
    CK_TILE_DEVICE void set_window_origin_extended(const BottomTensorIndex&) {}

    CK_TILE_DEVICE constexpr void
    set_bottom_tensor_view_data_ptr(typename BottomTensorView::DataType* data)
    {
        bottom_tensor_view_.buf_.p_data_ = data;
    }

    // move window-origin
    CK_TILE_DEVICE void move(const BottomTensorIndex& step)
    {
        window_origin_ += step;

        // Delegate to child if it implements extra movement logic
        static_cast<TileWindowType_*>(this)->move_extended(step);
    }

    // Default no-op; can be overridden in child
    CK_TILE_DEVICE void move_extended(const BottomTensorIndex&) {}
};

template <typename TileWindowType_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_>
struct tile_window_with_tile_dstr_base
    : public tile_window_base<TileWindowType_, BottomTensorView_, WindowLengths_>
{
    using TileDstr       = remove_cvref_t<StaticTileDistribution_>;
    using TileWindowBase = tile_window_base<TileWindowType_, BottomTensorView_, WindowLengths_>;

    using WindowAdaptor = typename TileDstr::PsYs2XsAdaptor;
    // using BottomTensorDesc = typename TileWindowBase::BottomTensorView::TensorDesc;
    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::get_num_of_top_dimension();

    static constexpr index_t NDimP = TileDstr::get_num_of_dimension_p();
    static constexpr index_t NDimY = TileDstr::get_num_of_dimension_y();

    using AdaptorTopIndex = array<index_t, NDimWindowAdaptorTop>;
    // using BottomTensorIndex = array<index_t, TileWindowBase::NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord = decltype(make_tensor_coordinate(
        typename TileWindowBase::BottomTensorDesc{}, typename TileWindowBase::BottomTensorIndex{}));

    static_assert(TileDstr::is_static(), "wrong!");
    static_assert(TileWindowBase::NDimBottomTensor == WindowAdaptor::get_num_of_bottom_dimension(),
                  "wrong! inconsistent # of diemsnions");

    CK_TILE_DEVICE constexpr auto get_tile_distribution() const { return tile_dstr_; }
    CK_TILE_HOST_DEVICE void init_raw() { this->bottom_tensor_view_.init_raw(); }

    CK_TILE_DEVICE static constexpr bool has_static_tile_distribution()
    {
        return TileDstr::is_static();
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    template <typename ATopIndex>
    CK_TILE_DEVICE void move_window_adaptor_and_bottom_tensor_thread_coordinate(
        WindowAdaptorCoord& window_adaptor_thread_coord,
        BottomTensorCoord& bottom_tensor_thread_coord,
        const ATopIndex& idx_diff_adaptor_top) const
    {
        array<index_t, TileWindowBase::NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                       window_adaptor_thread_coord,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(this->bottom_tensor_view_.get_tensor_descriptor(),
                               bottom_tensor_thread_coord,
                               idx_diff_adaptor_bottom);
    }

    // Tile tensor distribution, which contains:
    //   1. adaptor for window: [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    //   2. thread descriptor for thread tensor in register: [y0, y1, ...] ==> [d]
    TileDstr tile_dstr_;
};

} // namespace ck_tile
