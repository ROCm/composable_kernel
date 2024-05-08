// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"

namespace ck {
namespace tile_program {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename DataType_>
__device__ void
store_tile(TileWindowWithStaticLengths<BottomTensorView_, WindowLengths_>& tile_window_tmp,
           const StaticDistributedTensor<DataType_, TileDistribution_>& dstr_tensor)
{
    using DataType = remove_cvref_t<typename BottomTensorView_::DataType>;
    using TileDstr = remove_cvref_t<TileDistribution_>;

    static_assert(is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    auto tile_window = make_tile_window(tile_window_tmp.GetBottomTensorView(),
                                        tile_window_tmp.GetWindowLengths(),
                                        tile_window_tmp.GetWindowOrigin(),
                                        tile_dstr);

    tile_window.Store(dstr_tensor);
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename DataType_>
__device__ void
store_tile_raw(TileWindowWithStaticLengths<BottomTensorView_, WindowLengths_>& tile_window_tmp,
               const StaticDistributedTensor<DataType_, TileDistribution_>& dstr_tensor)
{
    using DataType = remove_cvref_t<typename BottomTensorView_::DataType>;
    using TileDstr = remove_cvref_t<TileDistribution_>;

    static_assert(is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    auto tile_window = make_tile_window(tile_window_tmp.GetBottomTensorView(),
                                        tile_window_tmp.GetWindowLengths(),
                                        tile_window_tmp.GetWindowOrigin(),
                                        tile_dstr);

    tile_window.StoreRaw(dstr_tensor);
}

} // namespace tile_program
} // namespace ck
