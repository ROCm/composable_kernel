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
    static constexpr index_t NDimBottomTensor = BottomTensorDesc::get_num_of_dimension();
    using BottomTensorIndex                   = array<index_t, NDimBottomTensor>;

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;

    // window_lengths
    using WindowLengths = remove_cvref_t<WindowLengths_>;
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
        static_cast<TileWindowType_*>(this)->set_window_origin_extra(new_window_origin);
    }
    // Default no-op; can be overridden in child
    CK_TILE_DEVICE void set_window_origin_extra(const BottomTensorIndex&) {}

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
        static_cast<TileWindowType_*>(this)->move_extra(step);
    }

    // Default no-op; can be overridden in child
    CK_TILE_DEVICE void move_extra(const BottomTensorIndex&) {}
};
} // namespace ck_tile
