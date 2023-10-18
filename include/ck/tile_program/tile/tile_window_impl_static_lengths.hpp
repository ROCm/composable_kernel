// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_adaptor_coordinate.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"

namespace ck {
namespace tile_program {

template <typename BottomTensorView_, typename WindowLengths_>
struct TileWindowWithStaticLengths
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;
    using DataType         = typename BottomTensorView::DataType;

    static constexpr index_t NDimBottomTensor = BottomTensorDesc::GetNumOfDimension();

    static_assert(is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");

    using BottomTensorIndex = Array<index_t, NDimBottomTensor>;

    __device__ constexpr TileWindowWithStaticLengths() = default;

    __device__ constexpr TileWindowWithStaticLengths(const BottomTensorView& bottom_tensor_view,
                                                     const WindowLengths& window_lengths,
                                                     const BottomTensorIndex& window_origin)
        : bottom_tensor_view_{bottom_tensor_view},
          window_lengths_{window_lengths},
          window_origin_{window_origin}
    {
    }

    __device__ static constexpr index_t GetNumOfDimension() { return NDimBottomTensor; }

    __device__ constexpr auto GetWindowLengths() const { return window_lengths_; }

    __device__ constexpr auto GetBottomTensorView() const { return bottom_tensor_view_; }

    __device__ constexpr auto GetWindowOrigin() const { return window_origin_; }

    // move window-origin
    __device__ void Move(const BottomTensorIndex& step) { window_origin_ += step; }

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    //
    WindowLengths window_lengths_;

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;
};

template <typename TensorView_, typename WindowLengths_>
__device__ constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const MultiIndex<TensorView_::GetNumOfDimension()>& origin)
{
    static_assert(is_known_at_compile_time<WindowLengths_>::value,
                  "wrong! lengths should be static");

    return TileWindowWithStaticLengths<remove_cvref_t<TensorView_>, remove_cvref_t<WindowLengths_>>{
        tensor_view, window_lengths, origin};
}

template <typename TensorView_, typename WindowLengths_>
__device__ void move_tile_window(
    TileWindowWithStaticLengths<TensorView_, WindowLengths_>& window,
    const typename TileWindowWithStaticLengths<TensorView_, WindowLengths_>::BottomTensorIndex&
        step)
{
    window.Move(step);
}

} // namespace tile_program
} // namespace ck
