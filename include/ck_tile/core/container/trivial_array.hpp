// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {
// trivial_array has no custom constructor, allowing it to be used as a constexpr variable type
template <typename T, index_t N>
struct trivial_array
{
    // Public aggregate initialization makes this a literal type
    T data[N];

    // Basic constexpr accessors
    constexpr const T& operator[](index_t i) const { return data[i]; }
    constexpr T& operator[](index_t i) { return data[i]; }

    constexpr static index_t size() { return N; }
};
} // namespace ck_tile
