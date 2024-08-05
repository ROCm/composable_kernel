// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"

namespace ck_tile {

template <typename DataType_, typename TensorViewLengths_, typename TensorViewStrides_>
struct SimpleTileWindowNavigator
{
    using DataType          = DataType_;
    using TensorViewLengths = TensorViewLengths_;
    using TensorViewStrides = TensorViewStrides_;

    CK_TILE_DEVICE constexpr SimpleTileWindowNavigator(const TensorViewLengths& lengths_,
                                                       const TensorViewStrides& strides_)
        : lengths(lengths_), strides(strides_)
    {
    }

    template <typename TensorView, typename WindowLengths>
    CK_TILE_DEVICE static constexpr auto
    make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const multi_index<TensorView::get_num_of_dimension()>& window_origin)
    {
        return ck_tile::make_tile_window(tile_window, window_origin);
    }

    template <typename TileWindow>
    CK_TILE_DEVICE void
    move_tile_window(TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step)
    {
        ck_tile::move_tile_window(tile_window, step);
    }

    TensorViewLengths lengths;
    TensorViewStrides strides;
};

template <typename DataType, typename TensorViewLengths, typename TensorViewStrides>
CK_TILE_DEVICE constexpr auto make_tile_window_navigator(const TensorViewLengths& lengths,
                                                         const TensorViewStrides& strides)
{
    return SimpleTileWindowNavigator<DataType, TensorViewLengths, TensorViewStrides>(lengths,
                                                                                     strides);
}

template <typename DataType_,
          index_t VirtualDim_,
          typename TensorViewLengths_,
          typename TensorViewStrides_>
struct PagedTileWindowNavigator
{
    using DataType                      = DataType_;
    static constexpr index_t VirtualDim = VirtualDim_;
    static_assert(VirtualDim == 0 || VirtualDim == 1);
    using TensorViewLengths = TensorViewLengths_;
    using TensorViewStrides = TensorViewStrides_;

    CK_TILE_DEVICE constexpr PagedTileWindowNavigator(copy_const_t<DataType, void>* blocks_,
                                                      long_index_t block_stride_,
                                                      long_index_t head_stride_,
                                                      long_index_t row_stride_,
                                                      const int32_t* block_indices_,
                                                      index_t num_blocks_,
                                                      index_t page_block_size_,
                                                      const TensorViewLengths& lengths_,
                                                      const TensorViewStrides& strides_)
        : blocks(reinterpret_cast<DataType*>(blocks_)),
          block_stride(block_stride_),
          head_stride(head_stride_),
          row_stride(row_stride_),
          block_indices(block_indices_),
          num_blocks(num_blocks_),
          page_block_size(page_block_size_),
          lengths(lengths_),
          strides(strides_)
    {
    }

    template <typename TensorView, typename WindowLengths>
    CK_TILE_DEVICE auto
    make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const multi_index<TensorView::get_num_of_dimension()>& window_origin) const
    {
        /// TODO: convert global window origin to local window origin
        auto local_window_origin = window_origin;

        return ck_tile::make_tile_window(tile_window, local_window_origin);
    }

    template <typename TileWindow>
    CK_TILE_DEVICE void
    move_tile_window(TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step) const
    {
        /// TODO: reset pointer and adjust local window origin
        ck_tile::move_tile_window(tile_window, step);
    }

    DataType* get_block_base(index_t block_index)
    {
        return blocks + block_index * block_stride + head_stride;
    }

    DataType* base(index_t i_virtual) { return get_block_base(); }

    DataType* blocks;
    long_index_t block_stride;
    long_index_t head_stride;
    long_index_t row_stride;

    const int32_t* block_indices;
    index_t num_blocks;
    index_t page_block_size;

    TensorViewLengths lengths;
    TensorViewStrides strides;
};

template <typename DataType,
          index_t VirtualDim,
          typename TensorViewLengths,
          typename TensorViewStrides>
CK_TILE_DEVICE constexpr auto make_tile_window_navigator(copy_const_t<DataType, void>* blocks,
                                                         long_index_t block_stride,
                                                         long_index_t head_stride,
                                                         long_index_t row_stride,
                                                         const int32_t* block_indices,
                                                         index_t num_blocks,
                                                         index_t page_block_size,
                                                         const TensorViewLengths& lengths,
                                                         const TensorViewStrides& strides)
{
    return PagedTileWindowNavigator<DataType, VirtualDim, TensorViewLengths, TensorViewStrides>(
        blocks,
        block_stride,
        head_stride,
        row_stride,
        block_indices,
        num_blocks,
        page_block_size,
        lengths,
        strides);
}

} // namespace ck_tile
