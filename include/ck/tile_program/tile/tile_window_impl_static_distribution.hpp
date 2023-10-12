// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_adaptor_coordinate.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"

namespace ck {
namespace tile_program {

template <typename BottomTensorView_, typename WindowLengths_, typename StaticTileDistribution_>
struct TileWindowWithStaticDistribution
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<StaticTileDistribution_>;

    using WindowAdaptor    = typename TileDstr::PsYs2XsAdaptor;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;

    using DataType = typename BottomTensorView::DataType;

    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::GetNumOfTopDimension();
    static constexpr index_t NDimBottomTensor     = BottomTensorDesc::GetNumOfDimension();

    // TODO: check WindowLengths and StaticTileDistribution are consistent

    static_assert(is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    static_assert(TileDstr::IsStatic(), "wrong!");

    static_assert(NDimBottomTensor == WindowAdaptor::GetNumOfBottomDimension(),
                  "wrong! inconsistent # of diemsnions");

    using AdaptorTopIndex   = Array<index_t, NDimWindowAdaptorTop>;
    using BottomTensorIndex = Array<index_t, NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    __device__ constexpr TileWindowWithStaticDistribution() = default;

    __device__ constexpr TileWindowWithStaticDistribution(
        const BottomTensorView& bottom_tensor_view,
        const WindowLengths& window_lengths,
        const BottomTensorIndex& window_origin,
        const TileDstr& tile_distribution)
        : bottom_tensor_view_{bottom_tensor_view},
          window_lengths_{window_lengths},
          window_origin_{window_origin},
          bottom_tensor_thread_coord_{},
          tile_dstr_{tile_distribution},
          window_adaptor_thread_coord_{}
    {
#if 0 // debug
      // only support warp-tile and block-tile
        static_assert(TileDstr::NDimP == 1 or TileDstr::NDimP == 2, "wrong!");

        if constexpr(TileDstr::NDimP == 1)
        {
            window_adaptor_thread_coord_ = make_tensor_adaptor_coordinate(
                tile_distribution.GetPsYs2XsAdaptor(), AdaptorTopIndex{get_lane_id(), 0});
        }
        else if constexpr(TileDstr::NDimP == 2)
        {
            window_adaptor_thread_coord_ =
                make_tensor_adaptor_coordinate(tile_distribution.GetPsYs2XsAdaptor(),
                                               AdaptorTopIndex{get_warp_id(), get_lane_id(), 0});
        }
#elif 0
        // only support warp-tile and block-tile
        static_assert(TileDstr::NDimP == 1 or TileDstr::NDimP == 2, "wrong!");

        if constexpr(TileDstr::NDimP == 1)
        {
            window_adaptor_thread_coord_ = make_tensor_adaptor_coordinate(
                tile_distribution.GetPsYs2XsAdaptor(),
                container_concat(Array<index_t, 1>{get_lane_id()},
                                 Array<index_t, TileDstr::NDimY>{0}));
        }
        else if constexpr(TileDstr::NDimP == 2)
        {
            window_adaptor_thread_coord_ = make_tensor_adaptor_coordinate(
                tile_distribution.GetPsYs2XsAdaptor(),
                container_concat(Array<index_t, 2>{get_warp_id(), get_lane_id()},
                                 Array<index_t, TileDstr::NDimY>{0}));
        }
#else
        window_adaptor_thread_coord_ = make_tensor_adaptor_coordinate(
            tile_distribution.GetPsYs2XsAdaptor(),
            container_concat(detail::get_partition_index(tile_distribution),
                             Array<index_t, TileDstr::NDimY>{0}));
#endif

        BottomTensorIndex bottom_tensor_thread_origin_idx;

        for(index_t i = 0; i < NDimBottomTensor; ++i)
        {
            bottom_tensor_thread_origin_idx(i) =
                window_origin[i] + window_adaptor_thread_coord_.GetBottomIndex()[i];
        }

        bottom_tensor_thread_coord_ = make_tensor_coordinate(
            bottom_tensor_view_.GetTensorDescriptor(), bottom_tensor_thread_origin_idx);
    }

    __device__ static constexpr index_t GetNumOfDimension() { return NDimBottomTensor; }

    __device__ static constexpr bool HasStaticTileDistribution() { return TileDstr::IsStatic(); }

    __device__ constexpr auto GetWindowLengths() const { return window_lengths_; }

    __device__ constexpr auto GetTileDistribution() const { return tile_dstr_; }

    __device__ constexpr auto GetBottomTensorView() const { return bottom_tensor_view_; }

    __device__ constexpr auto GetWindowOrigin() const { return window_origin_; }

    __device__ constexpr auto GetBottomTensorThreadCoordinate() const
    {
        return bottom_tensor_thread_coord_;
    }

    // move thread's window adaptor coordiante
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    __device__ void MoveWindowAdaptorThreadCoordinate(const AdaptorTopIndex& idx_diff_adaptor)
    {
        move_tensor_adaptor_coordinate(
            tile_dstr_.GetPsYs2XsAdaptor(), window_adaptor_thread_coord_, idx_diff_adaptor);
    }

    // move thread's botom tensor coordiante
    // [x0', x1', ... ] ==> [offset]
    __device__ void MoveBottomTensorThreadCoordinate(const BottomTensorIndex& idx_diff_tensor)
    {
        move_tensor_coordinate(bottom_tensor_view_.GetTensorDescriptor(),
                               bottom_tensor_thread_coord_,
                               idx_diff_tensor);
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    __device__ void
    MoveWindowAdaptorAndBottomTensorThreadCoordinate(const AdaptorTopIndex& idx_diff_adaptor_top)
    {
        Array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(tile_dstr_.GetPsYs2XsAdaptor(),
                                       window_adaptor_thread_coord_,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(bottom_tensor_view_.GetTensorDescriptor(),
                               bottom_tensor_thread_coord_,
                               idx_diff_adaptor_bottom);
    }

    // return vector dimension among [y0, y1, ...]
    __device__ static constexpr auto GetWindowAdaptorYsSafeVectorLengthStrides()
    {
        // bottom tensor top dimension vector lengths and strides
        const auto [bottom_tensor_top_dim_vector_lengths, bottom_tensor_top_dim_vector_strides] =
            BottomTensorDesc::GetTopDimensionSafeVectorLengthStrides();

        // window vector lengths/strides
        const auto window_adaptor_bottom_dim_vector_lengths = bottom_tensor_top_dim_vector_lengths;
        const auto window_adaptor_bottom_dim_vector_strides = bottom_tensor_top_dim_vector_strides;

        // window adaptor [p0, p1, ..., y0, y1, ...]
        Array<index_t, WindowAdaptor::GetNumOfHiddenDimension()> window_adaptor_vector_lengths{-1};
        Array<index_t, WindowAdaptor::GetNumOfHiddenDimension()> window_adaptor_vector_strides{-1};

        constexpr auto window_adaptor_bottom_dims = WindowAdaptor::GetBottomDimensionHiddenIds();

        set_container_subset(window_adaptor_vector_lengths,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_lengths);
        set_container_subset(window_adaptor_vector_strides,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_strides);

        const auto [window_adaptor_ps_ys_vector_lengths, window_adaptor_ps_ys_vector_strides] =
            WindowAdaptor{}.GetTopDimensionSafeVectorLengthStrides(window_adaptor_vector_lengths,
                                                                   window_adaptor_vector_strides);

        // [y0, y1, ...]
        constexpr auto y_dims = typename arithmetic_sequence_gen<TileDstr::GetNumOfDimensionP(),
                                                                 NDimWindowAdaptorTop,
                                                                 1>::type{};

        return make_tuple(get_container_subset(window_adaptor_ps_ys_vector_lengths, y_dims),
                          get_container_subset(window_adaptor_ps_ys_vector_strides, y_dims));
    }

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    //
    WindowLengths window_lengths_;

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;

    // per-thread coordinate for bottom tensor
    BottomTensorCoord bottom_tensor_thread_coord_;

    // Tile tensor distribution, which contains:
    //   1. adaptor for window: [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    //   2. thread descriptor for thread tensor in register: [y0, y1, ...] ==> [d]
    TileDstr tile_dstr_;

    //    thread window coordinate
    WindowAdaptorCoord window_adaptor_thread_coord_;
};

// TODO: use strategy
template <typename TensorView_, typename WindowLengths_, typename StaticTileDistribution_>
__device__ constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const MultiIndex<TensorView_::GetNumOfDimension()>& origin,
                 const StaticTileDistribution_& tile_distribution)
{
    return TileWindowWithStaticDistribution<remove_cvref_t<TensorView_>,
                                            remove_cvref_t<WindowLengths_>,
                                            remove_cvref_t<StaticTileDistribution_>>{
        tensor_view, window_lengths, origin, tile_distribution};
}

template <typename TensorView_, typename WindowLengths_, typename StaticTileDistribution_>
__device__ void move_tile_window(
    TileWindowWithStaticDistribution<TensorView_, WindowLengths_, StaticTileDistribution_>& window,
    const MultiIndex<
        TileWindowWithStaticDistribution<TensorView_, WindowLengths_, StaticTileDistribution_>::
            GetNumOfDimension()>& step)
{
    window.window_origin_ += step;

    window.MoveBottomTensorThreadCoordinate(step);
}

} // namespace tile_program
} // namespace ck
