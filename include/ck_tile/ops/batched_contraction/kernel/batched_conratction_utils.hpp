// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <ck_tile::index_t NumDimG>
CK_TILE_DEVICE constexpr auto DecomposeGIndex(ck_tile::index_t flat_g_idx,
                                              const ck_tile::index_t G_lengths[NumDimG])
{
    std::array<ck_tile::index_t, NumDimG> g_indices;

    auto remaining = flat_g_idx;
    static_for<0, NumDimG, 1>{}([&](auto i) {
        constexpr auto dim = NumDimG - 1 - i;
        g_indices[dim]     = remaining % G_lengths[dim];
        remaining          = remaining / G_lengths[dim];
    });
    return g_indices;
}

template <ck_tile::index_t NumDimG>
CK_TILE_DEVICE constexpr auto
CalculateGOffset(const std::array<ck_tile::index_t, NumDimG>& g_indices,
                 const ck_tile::index_t G_strides[NumDimG])
{
    ck_tile::index_t offset = 0;
    static_for<0, NumDimG, 1>{}([&](auto i) { offset += g_indices[i] * G_strides[i]; });
    return offset;
}

} // namespace ck_tile
