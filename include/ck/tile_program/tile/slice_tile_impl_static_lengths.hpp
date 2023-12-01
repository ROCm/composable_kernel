// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/tile_window_impl_static_lengths.hpp"

namespace ck {
namespace tile_program {

template <typename BottomTensorView_,
          typename WindowLengths_,
          index_t... SliceBegins,
          index_t... SliceEnds>
__host__ __device__ constexpr auto
get_slice_tile(const TileWindowWithStaticLengths<BottomTensorView_, WindowLengths_>& tile,
               Sequence<SliceBegins...> slice_begins,
               Sequence<SliceEnds...> slice_ends)
{
    using TileWindow = TileWindowWithStaticLengths<BottomTensorView_, WindowLengths_>;
    // NOTE: This API will override the origin of the tile window!
    static_assert(sizeof...(SliceBegins) == sizeof...(SliceEnds));
    static_assert(sizeof...(SliceBegins) == TileWindow::GetNumOfDimension());

    constexpr auto slice_lengths = slice_ends - slice_begins;

    return make_tile_window(tile.GetBottomTensorView(),
                            sequence_to_tuple_of_number(slice_lengths),
                            to_multi_index(slice_begins));
}

} // namespace tile_program
} // namespace ck
