// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/container/statically_indexed_array.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// this is 16 thread distribution
template <typename T, typename = void>
struct QuartTransposeDistribution;

template <typename T>
struct QuartTransposeDistribution<
    T,
    std::enable_if_t<std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, ck_tile::bf16_t>>>
{
    using TileDistribution = tile_distribution_encoding<sequence<>,
                                                        tuple<sequence<4>, sequence<4, 4>>,
                                                        tuple<sequence<1, 2>>,
                                                        tuple<sequence<0, 0>>,
                                                        sequence<2>,
                                                        sequence<1>>;
};

template <typename T>
struct QuartTransposeDistribution<
    T,
    std::enable_if_t<std::is_same_v<T, ck_tile::f8_t> || std::is_same_v<T, ck_tile::bf8_t>>>
{
    using TileDistribution = tile_distribution_encoding<sequence<>,
                                                        tuple<sequence<8>, sequence<2, 8>>,
                                                        tuple<sequence<1, 2>>,
                                                        tuple<sequence<0, 0>>,
                                                        sequence<2>,
                                                        sequence<1>>;
};

template <typename T, typename WarpLevelOuterDistribution_>
CK_TILE_HOST_DEVICE constexpr auto get_wavelevel_distribution()
{
    using WarpLevelDistribution = decltype(detail::make_embed_tile_distribution_encoding(
        WarpLevelOuterDistribution_{}, typename QuartTransposeDistribution<T>::TileDistribution{}));

    return WarpLevelDistribution{};
}

template <typename WarpLevelDistribution_, typename BlockLevelOuterDistribution_>
CK_TILE_HOST_DEVICE constexpr auto get_blocklevel_distribution()
{
    using BlockLevelDistribution = decltype(detail::make_embed_tile_distribution_encoding(
        BlockLevelOuterDistribution_{}, WarpLevelDistribution_{}));

    return BlockLevelDistribution{};
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto
load_tile_trans(const tile_window_with_static_distribution<BottomTensorView_,
                                                           WindowLengths_,
                                                           TileDistribution_,
                                                           NumCoord>& tile_window,
                number<i_access>                     = {},
                bool_constant<oob_conditional_check> = {})
{
}

} // namespace ck_tile
