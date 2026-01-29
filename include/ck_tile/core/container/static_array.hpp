// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {
// Fixed-size array with aggregate initialization
//
// This is a minimal array type designed for:
// - Constexpr/compile-time computation
// - GPU kernel code (trivially copyable)
// - Template metaprogramming
//
// Unlike ck_tile::array, this has no custom constructors,
// making it a literal type suitable for constexpr contexts.
// Use aggregate initialization: static_array<int, 3> arr{1, 2, 3};
template <typename T, index_t N>
struct static_array
{
    // Public aggregate initialization makes this a literal type
    T elems[N];

    // Basic constexpr accessors
    constexpr const T& operator[](index_t i) const { return elems[i]; }
    constexpr T& operator[](index_t i) { return elems[i]; }

    constexpr static index_t size() { return N; }
};
} // namespace ck_tile
