// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/tensor/tile_distribution_encoding.hpp"

namespace ck_tile {

// this generate wave level tile distribution
template <typename T, typename = void>
struct LaneGroupTransposeTraits;

template <typename T>
struct LaneGroupTransposeTraits<T, std::enable_if_t<sizeof(T) == 2>>
{
    // before transpose, 4x16
    static constexpr index_t ksecondDim = 4;
    static constexpr index_t kleadDim   = 16;
    // after transpose, 16x4
    static constexpr index_t ksecondDimT = 16;
    static constexpr index_t kleadDimT   = 4;
    using TileDistribution               = tile_distribution_encoding<sequence<>,
                                                        tuple<sequence<4>, sequence<4, 4>>,
                                                        tuple<sequence<1, 2>>,
                                                        tuple<sequence<0, 0>>,
                                                        sequence<2>,
                                                        sequence<1>>;
};

template <typename T>
struct LaneGroupTransposeTraits<T, std::enable_if_t<sizeof(T) == 1>>
{
    static constexpr index_t ksecondDim = 8;
    static constexpr index_t kleadDim   = 16;

    static constexpr index_t ksecondDimT = 16;
    static constexpr index_t kleadDimT   = 8;

    using TileDistribution = tile_distribution_encoding<sequence<>,
                                                        tuple<sequence<8>, sequence<2, 8>>,
                                                        tuple<sequence<1, 2>>,
                                                        tuple<sequence<0, 0>>,
                                                        sequence<2>,
                                                        sequence<1>>;
};

/*
 * @brief This function is used to generate the transposed distribution encoding
 *        for the given data type and distribution dimensions.
 *
 * @tparam T The data type of the elements in the tensor.
 * @tparam kOuterDistDim0 The outer distribution dimension 0, which is outer dimension for stride.
 * @tparam kOuterDistDim1 The outer distribution dimension 1, which is inner dimension for stride.
 * @tparam kInnerDistDim0 The inner distribution dimension 0, which is outer dimension for
 * consecutive.
 * @tparam kInnerDistDim1 The inner distribution dimension 1, which is inner dimension for
 * consecutive.
 */
template <typename T>
CK_TILE_DEVICE constexpr auto make_transposed_distr_encode()
{
    return typename LaneGroupTransposeTraits<T>::TileDistribution{};
}

} // namespace ck_tile
