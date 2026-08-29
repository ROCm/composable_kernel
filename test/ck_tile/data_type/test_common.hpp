// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include <cmath>
#include <limits>
#include <type_traits>

namespace ck_tile_test {

// ULP comparison utilities
// ULP = "Units in the Last Place".
// It measures the distance between two floating-point numbers as the number of representable values
// between them.
template <typename T>
inline uint64_t ulp_distance(T a, T b)
{
    static_assert(std::is_floating_point<T>::value, "ULP distance only for floating point types");

    if(std::isnan(a) || std::isnan(b))
        return std::numeric_limits<uint64_t>::max();
    if(std::isinf(a) || std::isinf(b))
    {
        if(a == b)
            return 0;
        return std::numeric_limits<uint64_t>::max();
    }

    // Use int32_t for float and int64_t for double
    using IntType  = std::conditional_t<sizeof(T) == 4, int32_t, int64_t>;
    using UIntType = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    IntType ia     = ck_tile::bit_cast<IntType>(a);
    IntType ib     = ck_tile::bit_cast<IntType>(b);

    // Make ia and ib lexicographically ordered as a twos-complement int
    // For float (32-bit): use 0x80000000, for double (64-bit): use 0x8000000000000000
    constexpr IntType sign_bit_mask =
        (sizeof(T) == 4) ? IntType(0x80000000) : IntType(0x8000000000000000LL);
    if(ia < 0)
        ia = sign_bit_mask - ia;
    if(ib < 0)
        ib = sign_bit_mask - ib;

    // Compute difference in unsigned type to avoid signed overflow (UB)
    UIntType ua = static_cast<UIntType>(ia);
    UIntType ub = static_cast<UIntType>(ib);
    return (ua > ub) ? (ua - ub) : (ub - ua);
}

} // namespace ck_tile_test
