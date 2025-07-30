// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

/// Declare a ck_tile::print() interface that gets specialized in each header file for types that
/// can be printed.
template <typename T>
CK_TILE_HOST_DEVICE void print(const T&)
{
    static_assert(sizeof(T) == 0,
                  "No print implementation available for this type. Please specialize "
                  "ck_tile::print for your type.");
}

/// Specialization for int
template <>
CK_TILE_HOST_DEVICE void print(const int& value)
{
    printf("%d", value);
}

/// Specialization for float
template <>
CK_TILE_HOST_DEVICE void print(const float& value)
{
    printf("%f", value);
}

} // namespace ck_tile
