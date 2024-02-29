// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/limits.hpp"
#include <hip/hip_fp16.h>

#pragma once

namespace ck_tile {

using fp16_hip_t = __half; // most of hip internal function use this type

CK_TILE_HOST_DEVICE
float fp16_to_float_hip(const fp16_hip_t& x);

CK_TILE_HOST_DEVICE
fp16_hip_t float_to_fp16_hip(const float& x);

// HIP use fp16_hip_t as interchangable data type for float16
struct alignas(2) half_t
{
    using raw_type = uint16_t;
    raw_type data;

    CK_TILE_HOST_DEVICE
    static constexpr half_t bit_cast(raw_type x)
    {
        half_t y;
        y.data = x;
        return y;
    }

    CK_TILE_HOST_DEVICE
    constexpr fp16_hip_t to_fp16() const { return ck_tile::bit_cast<fp16_hip_t>(data); }

    // constructor
    constexpr half_t() : data() {}

    // construct from HIP half
    CK_TILE_HOST_DEVICE
    explicit constexpr half_t(const fp16_hip_t& x) : data(ck_tile::bit_cast<raw_type>(x)) {}

    // construct from float
    CK_TILE_HOST_DEVICE
    explicit constexpr half_t(const float& x) : half_t(float_to_fp16_hip(x)) {}

    // construct from int
    CK_TILE_HOST_DEVICE
    explicit constexpr half_t(const int& x) : half_t(static_cast<fp16_hip_t>(__int2half_rn(x))) {}

    // construct from unsigned int
    CK_TILE_HOST_DEVICE
    explicit constexpr half_t(const unsigned int& x)
        : half_t(static_cast<fp16_hip_t>(__uint2half_rn(x)))
    {
    }

    // cast to float
    CK_TILE_HOST_DEVICE
    explicit constexpr operator float() const { return fp16_to_float_hip(to_fp16()); }

    // cast to int
    CK_TILE_HOST_DEVICE
    explicit constexpr operator int() const
    {
        return static_cast<int>(fp16_to_float_hip(to_fp16()));
    }

    // internal access
    CK_TILE_HOST_DEVICE
    constexpr raw_type& get() { return data; }

    CK_TILE_HOST_DEVICE
    constexpr raw_type get() const { return data; }
};

// conversions
CK_TILE_HOST_DEVICE
float fp16_to_float_hip(const fp16_hip_t& x)
{
    // return __half2float(x);
    return static_cast<float>(x);
}

CK_TILE_HOST_DEVICE
fp16_hip_t float_to_fp16_hip(const float& x)
{
    // return __float2half(x);
    return static_cast<fp16_hip_t>(x);
}

CK_TILE_HOST_DEVICE
float fp16_to_float(const half_t& x) { return static_cast<float>(x); }

CK_TILE_HOST_DEVICE
half_t float_to_fp16(const float& x) { return half_t{x}; }

// limits
template <class T>
struct numeric_limits;

template <>
struct numeric_limits<half_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr half_t min() { return half_t::bit_cast(0x0400); }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr half_t lowest() { return half_t::bit_cast(0xFBFF); }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr half_t max() { return half_t::bit_cast(0x7BFF); }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr half_t epsilon() { return half_t::bit_cast(0x1800); }

    // maximum rounding error
    CK_TILE_HOST_DEVICE static constexpr half_t round_error() { return half_t(0.5f); }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr half_t infinity() { return half_t::bit_cast(0x7C00); }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr half_t quiet_NaN() { return half_t::bit_cast(0x7FFF); }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr half_t signaling_NaN() { return half_t::bit_cast(0x7FFF); }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr half_t denorm_min() { return half_t::bit_cast(0x0001); }
};

template <typename T>
struct numeric_utils;

template <>
struct numeric_utils<half_t>
{
    static constexpr int exp            = 5;
    static constexpr int mant           = 10;
    static constexpr int bias           = 15;
    static constexpr uint16_t nan_mask  = 0x7C00;
    static constexpr uint16_t head_mask = 0xFC00;
    static constexpr uint16_t mant_mask = 0x3FF;
    static constexpr uint16_t exp_mask  = 0x1F;
    static constexpr uint32_t Inf       = 0x7C00;
    static constexpr uint32_t NegInf    = 0xFC00;
    static constexpr uint32_t NaN       = 0x7C01;
    static constexpr uint32_t Neg0      = 0x8000;
    using bitwise_type                  = uint16_t;
};

// arithmetic
CK_TILE_HOST_DEVICE
bool operator==(const half_t& x, const half_t& y) { return __heq(x.to_fp16(), y.to_fp16()); }

CK_TILE_HOST_DEVICE
bool operator!=(const half_t& x, const half_t& y) { return __hne(x.to_fp16(), y.to_fp16()); }

CK_TILE_HOST_DEVICE
bool operator<(const half_t& x, const half_t& y) { return __hlt(x.to_fp16(), y.to_fp16()); }

CK_TILE_HOST_DEVICE
bool operator<=(const half_t& x, const half_t& y) { return __hle(x.to_fp16(), y.to_fp16()); }

CK_TILE_HOST_DEVICE
bool operator>(const half_t& x, const half_t& y) { return __hgt(x.to_fp16(), y.to_fp16()); }

CK_TILE_HOST_DEVICE
bool operator>=(const half_t& x, const half_t& y) { return __hge(x.to_fp16(), y.to_fp16()); }

CK_TILE_HOST_DEVICE
half_t operator+(const half_t& x, const half_t& y)
{
    return half_t(__hadd(x.to_fp16(), y.to_fp16()));
}

CK_TILE_HOST_DEVICE
half_t operator-(const half_t& x) { return half_t(__hneg(x.to_fp16())); }

CK_TILE_HOST_DEVICE
half_t operator-(const half_t& x, const half_t& y)
{
    return half_t(__hsub(x.to_fp16(), y.to_fp16()));
}

CK_TILE_HOST_DEVICE
half_t operator*(const half_t& x, const half_t& y)
{
    return half_t(__hmul(x.to_fp16(), y.to_fp16()));
}

CK_TILE_HOST_DEVICE
half_t operator/(const half_t& x, const half_t& y)
{
    return half_t(__hdiv(x.to_fp16(), y.to_fp16()));
}

CK_TILE_HOST_DEVICE
half_t& operator+=(half_t& x, const half_t& y)
{
    x = half_t(__hadd(x.to_fp16(), y.to_fp16()));
    return x;
}

CK_TILE_HOST_DEVICE
half_t& operator-=(half_t& x, const half_t& y)
{
    x = half_t(__hsub(x.to_fp16(), y.to_fp16()));
    return x;
}

CK_TILE_HOST_DEVICE
half_t& operator*=(half_t& x, const half_t& y)
{
    x = half_t(__hmul(x.to_fp16(), y.to_fp16()));
    return x;
}

CK_TILE_HOST_DEVICE
half_t& operator/=(half_t& x, const half_t& y)
{
    x = half_t(__hdiv(x.to_fp16(), y.to_fp16()));
    return x;
}

CK_TILE_HOST_DEVICE
half_t& operator++(half_t& x)
{
    x = half_t(__hadd(x.to_fp16(), half_t(1.0f).to_fp16()));
    return x;
}

CK_TILE_HOST_DEVICE
half_t& operator--(half_t& x)
{
    x = half_t(__hsub(x.to_fp16(), half_t(1.0f).to_fp16()));
    return x;
}

CK_TILE_HOST_DEVICE
half_t operator++(half_t& x, int)
{
    half_t y(x);
    x = half_t(__hadd(x.to_fp16(), half_t(1.0f).to_fp16()));
    return y;
}

CK_TILE_HOST_DEVICE
half_t operator--(half_t& x, int)
{
    half_t y(x);
    x = half_t(__hsub(x.to_fp16(), half_t(1.0f).to_fp16()));
    return y;
}

// math
CK_TILE_HOST_DEVICE
half_t abs(const half_t& x) { return half_t::bit_cast(x.get() & 0x7fff); }

CK_TILE_HOST_DEVICE
bool isnan(const half_t& x)
{
    uint16_t xx = x.get();
    return (xx & 0x7FFF) > 0x7C00;
}

CK_TILE_DEVICE
half_t sqrt(half_t x)
{
    return static_cast<half_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x)));
};

CK_TILE_DEVICE
half_t exp(half_t x) { return static_cast<half_t>(__expf(static_cast<float>(x))); };

CK_TILE_DEVICE
half_t exp2(half_t x) { return static_cast<half_t>(exp2f(static_cast<float>(x))); };

CK_TILE_DEVICE
half_t log(half_t x) { return static_cast<half_t>(__logf(static_cast<float>(x))); };

using fp16_t = half_t;

} // namespace ck_tile
