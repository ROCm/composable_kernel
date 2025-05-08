// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

#if defined(__gfx950__)
// this generate wave level tile distribution
template <typename T, typename = void>
struct LaneGroupTransposeTraits;

template <typename T>
struct LaneGroupTransposeTraits<T, std::enable_if_t<sizeof(T) == 2>>
{
    template <index_t kOuterDistDim0,
              index_t kOuterDistDim1,
              index_t kInnerDistDim0,
              index_t kInnerDistDim1>
    using TileDistribution =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<kOuterDistDim0, kOuterDistDim1, 4>,
                                         sequence<kInnerDistDim0, kInnerDistDim1, 4, 4>>,
                                   tuple<sequence<1, 2, 1, 2>>,
                                   tuple<sequence<0, 0, 2, 2>>,
                                   sequence<2, 1, 2>,
                                   sequence<1, 1, 3>>;
};

template <typename T,
          index_t kOuterDistDim0,
          index_t kOuterDistDim1,
          index_t kInnerDistDim0,
          index_t kInnerDistDim1>
CK_TILE_DEVICE constexpr auto make_transposed_distr_encode()
{
    return LaneGroupTransposeTraits<
        T>::TileDistribution<kOuterDistDim0, kOuterDistDim1, kInnerDistDim0, kInnerDistDim1>{};
}

#endif
} // namespace ck_tile
