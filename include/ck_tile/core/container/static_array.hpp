// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck_tile {

/**
 * @brief Fixed-size array with aggregate initialization for constexpr contexts.
 *
 * Unlike ck_tile::array, this has no custom constructors, making it a literal type
 * suitable for constexpr evaluation and GPU kernel code. Use ck_tile::array when
 * constructors or non-trivial initialization are needed.
 * Use aggregate initialization: static_array<int, 3> arr{1, 2, 3};
 */
template <typename T, index_t N>
struct static_array
{
    // Public aggregate initialization makes this a literal type.
    // N == 0 uses size 1 to avoid zero-length arrays (non-standard).
    T elems[N > 0 ? N : 1];

    // Basic constexpr accessors
    CK_TILE_HOST_DEVICE constexpr const T& operator[](index_t i) const [[clang::lifetimebound]]
    {
        return elems[i];
    }
    CK_TILE_HOST_DEVICE constexpr T& operator[](index_t i) [[clang::lifetimebound]]
    {
        return elems[i];
    }

    CK_TILE_HOST_DEVICE static constexpr index_t size() { return N; }
};
} // namespace ck_tile

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
