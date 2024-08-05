// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"

namespace ck_tile {

template <typename DataType_>
struct SimpleTileWindowNavigator
{
    using DataType = DataType_;

    template <typename TensorView, typename WindowLengths>
    CK_TILE_DEVICE static constexpr auto
    make_tile_window(const TensorView& tensor_view,
                     const WindowLengths& window_lengths,
                     const multi_index<TensorView::get_num_of_dimension()>& window_origin)
    {
        return ck_tile::make_tile_window(tensor_view, window_lengths, window_origin);
    }

    template <typename TensorView, typename WindowLengths>
    CK_TILE_DEVICE static constexpr auto
    make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const multi_index<TensorView::get_num_of_dimension()>& window_origin)
    {
        return ck_tile::make_tile_window(tile_window, window_origin);
    }

    template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
    CK_TILE_DEVICE constexpr auto
    make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const StaticTileDistribution& tile_distribution)
    {
        return ck_tile::make_tile_window(tile_window, tile_distribution);
    }

    template <typename TileWindow>
    CK_TILE_DEVICE void
    move_tile_window(TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step)
    {
        ck_tile::move_tile_window(tile_window, step);
    }
};

template <typename DataType_, index_t VirtualDim_>
struct PagedTileWindowNavigator
{
    using DataType                      = DataType_;
    static constexpr index_t VirtualDim = VirtualDim_;
    static_assert(VirtualDim == 0 || VirtualDim == 1);

    CK_TILE_DEVICE constexpr PagedTileWindowNavigator(copy_const_t<DataType, void>* blocks_,
                                                      long_index_t block_stride_,
                                                      long_index_t head_stride_,
                                                      long_index_t row_stride_,
                                                      const int32_t* block_indices_,
                                                      index_t num_blocks_,
                                                      index_t page_block_size_)
        : blocks(reinterpret_cast<DataType*>(blocks_)),
          block_stride(block_stride_),
          head_stride(head_stride_),
          row_stride(row_stride_),
          block_indices(block_indices_),
          num_blocks(num_blocks_),
          page_block_size(page_block_size_)
    {
    }

    template <typename TensorView, typename WindowLengths>
    CK_TILE_DEVICE auto
    make_tile_window(const TensorView& tensor_view,
                     const WindowLengths& window_lengths,
                     const multi_index<TensorView::get_num_of_dimension()>& window_origin) const
    {
        auto tile_window = ck_tile::make_tile_window(tensor_view, window_lengths, window_origin);
        /// TODO: convert global window origin to local window origin
        return tile_window;
    }

    template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
    CK_TILE_DEVICE auto
    make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const StaticTileDistribution& tile_distribution) const
    {
        auto new_tile_window = ck_tile::make_tile_window(tile_window, tile_distribution);
        /// TODO: convert global window origin to local window origin
        return new_tile_window;
    }

    template <typename TensorView, typename WindowLengths>
    CK_TILE_DEVICE auto
    make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const multi_index<TensorView::get_num_of_dimension()>& window_origin) const
    {
        auto new_tile_window = ck_tile::make_tile_window(tile_window, window_origin);
        /// TODO: convert global window origin to local window origin
        return new_tile_window;
    }

    template <typename TileWindow>
    CK_TILE_DEVICE void
    move_tile_window(TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step) const
    {
        ck_tile::move_tile_window(tile_window, step);
    }

    private:
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
};

} // namespace ck_tile
