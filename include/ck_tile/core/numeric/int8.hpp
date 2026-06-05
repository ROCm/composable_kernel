// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"

#include <cstdint>

namespace ck_tile {

// use int8_t directly for int8 arithemetic
// here one can use ck_tile::int8_t to access original int8_t
using int8_t = std::int8_t;

// limits
template <class T>
struct numeric;

template <>
struct numeric<int8_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr int8_t min() { return int8_t(-128); }

    // minimum finite value
    CK_TILE_HOST_DEVICE static constexpr int8_t lowest() { return int8_t(-128); }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr int8_t max() { return int8_t(127); }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr int8_t epsilon()
    {
        return 1; // not used
    }

    CK_TILE_HOST_DEVICE static constexpr int8_t round_error()
    {
        return 1; // not used
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr int8_t infinity()
    {
        return 1; // not used
    }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr int8_t quiet_NaN()
    {
        return 1; // not used
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr int8_t signaling_NaN()
    {
        return 1; // not used
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr int8_t denorm_min()
    {
        return 1; // not used
    }

    CK_TILE_HOST_DEVICE static constexpr int8_t zero() { return 0; }
};

CK_TILE_HOST_DEVICE
constexpr float int8_to_float(const int8_t& x) { return static_cast<float>(x); }

CK_TILE_HOST_DEVICE
constexpr int8_t float_to_int8(const float& x) { return static_cast<int8_t>(x); }

#if !CK_TILE_USE_CUSTOM_DATA_TYPE
#define CK_TILE_TYPE_CONVERT(dtype_, dname_, stype_, sname_)                    \
    template <>                                                                 \
    CK_TILE_HOST_DEVICE constexpr dtype_ type_convert<dtype_, stype_>(stype_ x) \
    {                                                                           \
        return sname_##_to_##dname_(x);                                         \
    }

CK_TILE_TYPE_CONVERT(float, float, int8_t, int8)
CK_TILE_TYPE_CONVERT(int8_t, int8, float, float)

#undef CK_TILE_TYPE_CONVERT
#endif

} // namespace ck_tile
