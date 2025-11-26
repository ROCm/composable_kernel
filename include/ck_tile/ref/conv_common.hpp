// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_TILE_REF_CONV_COMMON_HPP
#define CK_TILE_REF_CONV_COMMON_HPP

#include "ck_tile/core.hpp"
#include <array>
#include <vector>

namespace ck_tile {

// Helper function to convert std::vector to std::array for kernel parameters
template <ck_tile::index_t NDimSpatial>
inline std::array<ck_tile::long_index_t, NDimSpatial>
to_array(const std::vector<ck_tile::long_index_t>& vec)
{
    std::array<ck_tile::long_index_t, NDimSpatial> arr;
    for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
    {
        arr[i] = vec[i];
    }
    return arr;
}

// Helper to fill missing dimensions with default value
template <ck_tile::index_t NDimSpatial>
inline std::array<ck_tile::long_index_t, NDimSpatial>
to_array_with_default(const std::vector<ck_tile::long_index_t>& vec,
                      ck_tile::long_index_t default_val = 1)
{
    std::array<ck_tile::long_index_t, NDimSpatial> arr;
    for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
    {
        arr[i] = (static_cast<size_t>(i) < vec.size()) ? vec[i] : default_val;
    }
    return arr;
}

} // namespace ck_tile

#endif
