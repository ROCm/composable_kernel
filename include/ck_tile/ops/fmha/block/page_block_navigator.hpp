// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"

namespace ck_tile {

// assume that we have only 1 page-block
template <typename DataType_>
struct TrivialPageBlockNavigator
{
    using DataType     = DataType_;
    using WindowOrigin = multi_index<2>;

    template <typename TensorView, typename WindowLengths>
    CK_TILE_DEVICE static constexpr auto
    make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const WindowOrigin& window_origin)
    {
        return make_tuple(
            /*block_index=*/0, ck_tile::make_tile_window(tile_window, window_origin));
    }

    template <typename TensorView, typename WindowLengths, typename TileDistribution>
    CK_TILE_DEVICE static constexpr auto
    make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const WindowOrigin& window_origin,
                     const TileDistribution& tile_distribution)
    {
        return make_tuple(
            /*block_index=*/0,
            ck_tile::make_tile_window(tile_window, window_origin, tile_distribution));
    }

    template <typename TileWindow>
    CK_TILE_DEVICE static index_t
    move_tile_window(index_t /*block_index*/,
                     TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step)
    {
        ck_tile::move_tile_window(tile_window, step);

        return /*block_index=*/0;
    }

    /// TODO: remove this method after finish debuging
    CK_TILE_DEVICE static constexpr int32_t
    get_block_index(const WindowOrigin& /*global_window_origin*/)
    {
        return /*block_index=*/0;
    }

    CK_TILE_DEVICE static constexpr WindowOrigin
    to_local_window_origin(const WindowOrigin& global_window_origin)
    {
        return global_window_origin;
    }

    CK_TILE_DEVICE static constexpr WindowOrigin
    to_global_window_origin(index_t /*block_index*/, const WindowOrigin& local_window_origin)
    {
        return local_window_origin;
    }
};

// default page-block navigator, assume that tensor view size is same as page-block size
template <typename DataType_, index_t VirtualDim_>
struct PageBlockNavigator
{
    using DataType                      = DataType_;
    static constexpr index_t VirtualDim = VirtualDim_;
    static_assert(VirtualDim == 0 || VirtualDim == 1, "only support 2d tile window");
    using WindowOrigin = multi_index<2>;

    CK_TILE_HOST_DEVICE constexpr PageBlockNavigator(copy_const_t<DataType, void>* physical_blocks_,
                                                     long_index_t block_stride_,
                                                     long_index_t fixed_offset_,
                                                     const int32_t* physical_block_indices_,
                                                     index_t num_blocks_,
                                                     index_t page_block_size_)
        : physical_blocks(reinterpret_cast<DataType*>(physical_blocks_)),
          block_stride(block_stride_),
          fixed_offset(fixed_offset_),
          physical_block_indices(physical_block_indices_),
          num_blocks(num_blocks_),
          page_block_size(page_block_size_)
    {
    }

    template <typename TensorView, typename WindowLengths>
    CK_TILE_HOST_DEVICE auto
    make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const WindowOrigin& window_origin) const
    {
        const index_t block_index              = get_block_index(window_origin);
        const WindowOrigin local_window_origin = to_local_window_origin(window_origin);

        auto new_tile_window = ck_tile::make_tile_window(tile_window, local_window_origin);
        new_tile_window.set_bottom_tensor_view_data_ptr(get_block_ptr(block_index));

        return make_tuple(block_index, new_tile_window);
    }

    template <typename TensorView, typename WindowLengths, typename TileDistribution>
    CK_TILE_HOST_DEVICE auto
    make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const WindowOrigin& window_origin,
                     const TileDistribution& tile_distribution) const
    {
        const index_t block_index              = get_block_index(window_origin);
        const WindowOrigin local_window_origin = to_local_window_origin(window_origin);

        auto new_tile_window =
            ck_tile::make_tile_window(tile_window, local_window_origin, tile_distribution);
        new_tile_window.set_bottom_tensor_view_data_ptr(get_block_ptr(block_index));

        return make_tuple(block_index, new_tile_window);
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE index_t
    move_tile_window(index_t block_index,
                     TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step) const
    {

        ck_tile::move_tile_window(tile_window, step);

        const WindowOrigin global_window_origin =
            to_global_window_origin(block_index, tile_window.get_window_origin());
        const WindowOrigin local_window_origin = to_local_window_origin(global_window_origin);

        const index_t new_block_index = get_block_index(global_window_origin);
        tile_window.set_window_origin(local_window_origin);
        tile_window.set_bottom_tensor_view_data_ptr(get_block_ptr(new_block_index));

        return new_block_index;
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE bool is_cross_block(index_t block_index,
                                            const TileWindow& tile_window) const
    {
        const index_t origin = tile_window.get_window_origin().at(number<VirtualDim>{});
        const index_t length = tile_window.get_window_lengths().at(number<VirtualDim>{});
        return (block_index < num_blocks - 1) && (page_block_size < origin + length);
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE void
    move_to_block(index_t block_index, TileWindow& tile_window, index_t new_block_index) const
    {
        const multi_index<2> step = [&]() {
            const index_t origin_diff = (block_index - new_block_index) * page_block_size;
            if constexpr(VirtualDim == 0)
            {
                return make_multi_index(origin_diff, 0);
            }
            else
            {
                return make_multi_index(0, origin_diff);
            }
        }();

        tile_window.set_window_origin(tile_window.get_window_origin() + step);
        tile_window.set_bottom_tensor_view_data_ptr(get_block_ptr(new_block_index));
    }

    CK_TILE_HOST_DEVICE
    DataType* get_block_ptr(index_t block_index) const
    {
        return physical_blocks + physical_block_indices[block_index] * block_stride + fixed_offset;
    }

    CK_TILE_HOST_DEVICE int32_t get_block_index(const WindowOrigin& global_window_origin) const
    {
        return integer_divide_floor(global_window_origin.at(number<VirtualDim>{}), page_block_size);
    }

    CK_TILE_HOST_DEVICE WindowOrigin
    to_local_window_origin(const WindowOrigin& global_window_origin) const
    {
        if constexpr(VirtualDim == 0)
        {
            const index_t length              = global_window_origin.at(number<0>{});
            const index_t num_complete_blocks = integer_divide_floor(length, page_block_size);
            return make_multi_index(length - page_block_size * num_complete_blocks,
                                    global_window_origin.at(number<1>{}));
        }
        else
        {
            const index_t length              = global_window_origin.at(number<1>{});
            const index_t num_complete_blocks = integer_divide_floor(length, page_block_size);
            return make_multi_index(global_window_origin.at(number<0>{}),
                                    length - page_block_size * num_complete_blocks);
        }
    }

    CK_TILE_HOST_DEVICE WindowOrigin
    to_global_window_origin(index_t block_index, const WindowOrigin& local_window_origin) const
    {
        if constexpr(VirtualDim == 0)
        {
            return make_multi_index(block_index * page_block_size +
                                        local_window_origin.at(number<0>{}),
                                    local_window_origin.at(number<1>{}));
        }
        else
        {
            return make_multi_index(local_window_origin.at(number<0>{}),
                                    block_index * page_block_size +
                                        local_window_origin.at(number<1>{}));
        }
    }

    DataType* physical_blocks;
    long_index_t block_stride;
    long_index_t fixed_offset;

    const int32_t* physical_block_indices;
    index_t num_blocks;
    index_t page_block_size;
};

} // namespace ck_tile
