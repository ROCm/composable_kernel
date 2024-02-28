// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include <limits>
#include <stdint.h>

namespace ck_tile {

template <typename T>
struct numeric_limits
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr T min() { return std::numeric_limits<T>::min(); }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr T lowest() { return std::numeric_limits<T>::lowest(); }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr T max() { return std::numeric_limits<T>::max(); }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr T epsilon() { return std::numeric_limits<T>::epsilon(); }

    // maximum rounding error
    CK_TILE_HOST_DEVICE static constexpr T round_error()
    {
        return std::numeric_limits<T>::round_error();
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr T infinity() { return std::numeric_limits<T>::infinity(); }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr T quiet_NaN()
    {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr T signaling_NaN()
    {
        return std::numeric_limits<T>::signaling_NaN();
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr T denorm_min()
    {
        return std::numeric_limits<T>::denorm_min();
    }
};

template <typename T>
struct numeric_utils;

template <>
struct numeric_utils<float>
{
    static constexpr int exp            = 8;
    static constexpr int mant           = 23;
    static constexpr int bias           = 127;
    static constexpr uint32_t nan_mask  = 0x7F800000;
    static constexpr uint32_t head_mask = 0xFF800000;
    static constexpr uint32_t mant_mask = 0x7FFFFF;
    static constexpr uint32_t exp_mask  = 0xFF;
    static constexpr uint32_t Inf       = 0x7F800000;
    static constexpr uint32_t NegInf    = 0xFF800000;
    static constexpr uint32_t NaN       = 0x7F800001;
    static constexpr uint32_t Neg0      = 0x80000000;
    using bitwise_type                  = uint32_t;
};

} // namespace ck_tile
