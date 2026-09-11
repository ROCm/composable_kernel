// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/tensor/tile_window_linear.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// Reconstructs tile_window_with_static_distribution on every call, pre-computing XOR
// address coordinates before storing. For one-shot stores this is fine. For repeated
// stores in a hot loop, prefer constructing the distributed window once via
// make_tile_window(view, lengths, origin, dstr), then calling .store() directly.
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename DataType_>
CK_TILE_DEVICE void
store_tile(tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>& tile_window_tmp,
           const static_distributed_tensor<DataType_, TileDistribution_>& dstr_tensor)
{
    using DataType = remove_cvref_t<typename BottomTensorView_::DataType>;
    using TileDstr = remove_cvref_t<TileDistribution_>;

    static_assert(std::is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    auto tile_window = make_tile_window(tile_window_tmp.get_bottom_tensor_view(),
                                        tile_window_tmp.get_window_lengths(),
                                        tile_window_tmp.get_window_origin(),
                                        tile_dstr);

    tile_window.store(dstr_tensor);
}

// Same as above, but the caller supplies the partition index explicitly instead of
// deriving it from the hardware thread ID. Use this when a kernel manages multiple
// logical partitions per physical thread (e.g. double-buffer ping-pong) or when
// the default hardware-derived index is wrong for the current scheduling scheme.
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename DataType_>
CK_TILE_DEVICE void
store_tile(tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>& tile_window_tmp,
           const static_distributed_tensor<DataType_, TileDistribution_>& dstr_tensor,
           decltype(get_partition_index(dstr_tensor.get_tile_distribution())) partition_index)
{
    using DataType = remove_cvref_t<typename BottomTensorView_::DataType>;
    using TileDstr = remove_cvref_t<TileDistribution_>;

    static_assert(std::is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    auto tile_window = make_tile_window(tile_window_tmp.get_bottom_tensor_view(),
                                        tile_window_tmp.get_window_lengths(),
                                        tile_window_tmp.get_window_origin(),
                                        tile_dstr,
                                        partition_index);

    tile_window.store(dstr_tensor);
}

// Raw variant -- same reconstruction cost as store_tile above.
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename DataType_>
CK_TILE_DEVICE void
store_tile_raw(tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>& tile_window_tmp,
               const static_distributed_tensor<DataType_, TileDistribution_>& dstr_tensor)
{
    using DataType = remove_cvref_t<typename BottomTensorView_::DataType>;
    using TileDstr = remove_cvref_t<TileDistribution_>;

    static_assert(std::is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    auto tile_window = make_tile_window(tile_window_tmp.get_bottom_tensor_view(),
                                        tile_window_tmp.get_window_lengths(),
                                        tile_window_tmp.get_window_origin(),
                                        tile_dstr);

    tile_window.store_raw(dstr_tensor);
}

// Same as above, but the caller supplies the partition index explicitly instead of
// deriving it from the hardware thread ID. Use this when a kernel manages multiple
// logical partitions per physical thread (e.g. double-buffer ping-pong) or when
// the default hardware-derived index is wrong for the current scheduling scheme.
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename DataType_>
CK_TILE_DEVICE void
store_tile_raw(tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>& tile_window_tmp,
               const static_distributed_tensor<DataType_, TileDistribution_>& dstr_tensor,
               decltype(get_partition_index(dstr_tensor.get_tile_distribution())) partition_index)
{
    using DataType = remove_cvref_t<typename BottomTensorView_::DataType>;
    using TileDstr = remove_cvref_t<TileDistribution_>;

    static_assert(std::is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    auto tile_window = make_tile_window(tile_window_tmp.get_bottom_tensor_view(),
                                        tile_window_tmp.get_window_lengths(),
                                        tile_window_tmp.get_window_origin(),
                                        tile_dstr,
                                        partition_index);

    tile_window.store_raw(dstr_tensor);
}

// Uses pre-computed coordinates from the distributed window's construction.
// No coordinate reconstruction -- direct buffer stores via pre_computed_coords_.
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          typename DataType_>
CK_TILE_DEVICE void
store_tile(tile_window_with_static_distribution<BottomTensorView_,
                                                WindowLengths_,
                                                TileDistribution_,
                                                NumCoord>& tile_window,
           const static_distributed_tensor<DataType_, TileDistribution_>& dstr_tensor)
{
    tile_window.store(dstr_tensor, number<-1>{});
}

// Raw variant -- same fast path as above.
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          typename DataType_>
CK_TILE_DEVICE void
store_tile_raw(tile_window_with_static_distribution<BottomTensorView_,
                                                    WindowLengths_,
                                                    TileDistribution_,
                                                    NumCoord>& tile_window,
               const static_distributed_tensor<DataType_, TileDistribution_>& dstr_tensor)
{
    tile_window.store_raw(dstr_tensor, number<-1>{});
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          typename DataType_>
CK_TILE_DEVICE void store_tile(
    tile_window_linear<BottomTensorView_, WindowLengths_, TileDistribution_, LinearBottomDims_>&
        tile_window,
    const static_distributed_tensor<DataType_, TileDistribution_>& dstr_tensor)
{
    tile_window.store(dstr_tensor, number<-1>{});
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          typename DataType_>
CK_TILE_DEVICE void store_tile_raw(
    tile_window_linear<BottomTensorView_, WindowLengths_, TileDistribution_, LinearBottomDims_>&
        tile_window,
    const static_distributed_tensor<DataType_, TileDistribution_>& dstr_tensor)
{
    tile_window.store_raw(dstr_tensor, number<-1>{});
}

template <typename TDMConfig_,
          typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access = -1>
CK_TILE_DEVICE auto store_tile_tdm(const TDMConfig_& tdm_config,
                                   tile_window_with_static_distribution<BottomTensorView_,
                                                                        WindowLengths_,
                                                                        TileDistribution_,
                                                                        NumCoord>& tile_window,
                                   const LdsTileWindow_& lds_tile,
                                   number<i_access> = {})
{
    return tile_window.tdm_store_from_lds(tdm_config, lds_tile, number<i_access>{});
}

template <typename TDMConfig_,
          typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          index_t i_access = -1>
CK_TILE_DEVICE void
store_tile_tdm(const TDMConfig_& tdm_config,
               tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>& tile_window_tmp,
               const LdsTileWindow_& lds_tile,
               number<i_access> = {})
{
    using DataType    = remove_cvref_t<typename BottomTensorView_::DataType>;
    using LdsDataType = remove_cvref_t<typename remove_cvref_t<LdsTileWindow_>::DataType>;
    using TileDstr    = remove_cvref_t<typename remove_cvref_t<LdsTileWindow_>::TileDstr>;

    static_assert(std::is_same_v<remove_cvref_t<DataType>, LdsDataType>, "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    auto tile_window = make_tile_window(tile_window_tmp.get_bottom_tensor_view(),
                                        tile_window_tmp.get_window_lengths(),
                                        tile_window_tmp.get_window_origin(),
                                        tile_dstr);

    store_tile_tdm(tdm_config, tile_window, lds_tile, number<i_access>{});
}

} // namespace ck_tile
