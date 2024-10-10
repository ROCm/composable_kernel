// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/tile_window_linear.hpp"
#include "ck_tile/core/tensor/null_tile_window.hpp"
#include "ck_tile/core/tensor/null_tensor.hpp"

namespace ck_tile {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile(const tile_window_with_static_distribution<BottomTensorView_,
                                                                         WindowLengths_,
                                                                         TileDistribution_,
                                                                         NumCoord>& tile_window,
                              number<i_access>                     = {},
                              bool_constant<oob_conditional_check> = {})
{
    return tile_window.load(number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile(const tile_window_linear<BottomTensorView_,
                                                       WindowLengths_,
                                                       TileDistribution_,
                                                       LinearBottomDims_>& tile_window,
                              number<i_access>                     = {},
                              bool_constant<oob_conditional_check> = {})
{
    return tile_window.load(number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename T,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto load_tile_raw(T& tile,
                                  const tile_window_with_static_distribution<BottomTensorView_,
                                                                             WindowLengths_,
                                                                             TileDistribution_,
                                                                             NumCoord>& tile_window,
                                  number<i_access>                     = {},
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {})
{
    tile_window.load_raw(
        tile, number<i_access>{}, bool_constant<oob_conditional_check>{}, bool_constant<pre_nop>{});
}

template <typename T,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto load_tile_raw(T& tile,
                                  const tile_window_linear<BottomTensorView_,
                                                           WindowLengths_,
                                                           TileDistribution_,
                                                           LinearBottomDims_>& tile_window,
                                  number<i_access>                     = {},
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {})
{
    tile_window.load_raw(
        tile, number<i_access>{}, bool_constant<oob_conditional_check>{}, bool_constant<pre_nop>{});
}

// for this API we force user to use CK_TILE_LDS_ADDR attribute specified smem
// while creating the smem window, which can enable compiler properly detect the
// dependency if using multiple smem window (multiple buffer)
template <typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto
async_load_tile(LdsTileWindow_&& lds_tile,
                const tile_window_with_static_distribution<BottomTensorView_,
                                                           WindowLengths_,
                                                           TileDistribution_,
                                                           NumCoord>& tile_window,
                number<i_access>                     = {},
                bool_constant<oob_conditional_check> = {})
{
    return tile_window.async_load(
        lds_tile, number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto async_load_tile(LdsTileWindow_&& lds_tile,
                                    const tile_window_linear<BottomTensorView_,
                                                             WindowLengths_,
                                                             TileDistribution_,
                                                             LinearBottomDims_>& tile_window,
                                    number<i_access>                     = {},
                                    bool_constant<oob_conditional_check> = {})
{
    return tile_window.async_load(
        lds_tile, number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto
async_load_tile_raw(LdsTileWindow_&& lds_tile,
                    const tile_window_with_static_distribution<BottomTensorView_,
                                                               WindowLengths_,
                                                               TileDistribution_,
                                                               NumCoord>& tile_window,
                    number<i_access>                     = {},
                    bool_constant<oob_conditional_check> = {},
                    bool_constant<pre_nop>               = {})
{
    return tile_window.async_load_raw(lds_tile,
                                      number<i_access>{},
                                      bool_constant<oob_conditional_check>{},
                                      bool_constant<pre_nop>{});
}

template <typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto async_load_tile_raw(LdsTileWindow_&& lds_tile,
                                        const tile_window_linear<BottomTensorView_,
                                                                 WindowLengths_,
                                                                 TileDistribution_,
                                                                 LinearBottomDims_>& tile_window,
                                        number<i_access>                     = {},
                                        bool_constant<oob_conditional_check> = {},
                                        bool_constant<pre_nop>               = {})
{
    return tile_window.async_load_raw(lds_tile,
                                      number<i_access>{},
                                      bool_constant<oob_conditional_check>{},
                                      bool_constant<pre_nop>{});
}

template <typename WindowLengths, index_t i_access = -1, bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile(const null_tile_window<WindowLengths>&,
                              number<i_access>                     = {},
                              bool_constant<oob_conditional_check> = {})
{
    return null_tensor{};
}

template <typename T,
          typename WindowLengths,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto load_tile_raw(T& /*null_tile*/,
                                  const null_tile_window<WindowLengths>&,
                                  number<i_access>                     = {},
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {})
{
}

// TODO: this function requires some sub-fileds exist for the target tile window
template <typename TileWindow,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto load_tile_raw(const TileWindow& w,
                                  number<i_access>                     = {},
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {})
{
    using TileDstr = typename TileWindow::TileDstr;
    using DataType = typename TileWindow::DataType;

    auto t = make_static_distributed_tensor<DataType>(TileDstr{});

    load_tile_raw(
        t, w, number<i_access>{}, bool_constant<oob_conditional_check>{}, bool_constant<pre_nop>{});

    return t;
}

} // namespace ck_tile
