// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/numeric/arithmetic.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/limits.hpp"
#include <stdint.h>

#pragma once

namespace ck_tile {

enum class bf16_rounding_mode
{
    standard = 0, // rtn
    truncate_with_nan,
    truncate,
};

template <bf16_rounding_mode rounding =
              static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
CK_TILE_HOST_DEVICE uint16_t float_to_bf16_raw(float f, constant<rounding> = {});

template <bf16_rounding_mode rounding =
              static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
CK_TILE_HOST_DEVICE uint16_t double_to_bf16_raw(double f, constant<rounding> = {});

CK_TILE_HOST_DEVICE
float bf16_to_float_raw(uint16_t x);

CK_TILE_HOST_DEVICE
double bf16_to_double_raw(uint16_t x);

// HIP use __hip_bfloat16 as struct
struct alignas(2) bfloat16_t
{
    using raw_type = uint16_t;
    raw_type data;

    CK_TILE_HOST_DEVICE
    static constexpr bfloat16_t bit_cast(raw_type x)
    {
        bfloat16_t y;
        y.data = x;
        return y;
    }

    // constructor
    constexpr bfloat16_t() : data() {}

    // construct from float
    CK_TILE_HOST_DEVICE
    explicit constexpr bfloat16_t(const float& x) : data(float_to_bf16_raw(x)) {}

    // construct from double
    CK_TILE_HOST_DEVICE
    explicit constexpr bfloat16_t(const double& x) : data(double_to_bf16_raw(x)) {}

    // construct from int
    CK_TILE_HOST_DEVICE
    explicit constexpr bfloat16_t(const int& x) : data(float_to_bf16_raw(static_cast<float>(x))) {}

    // construct from unsigned int
    CK_TILE_HOST_DEVICE
    explicit constexpr bfloat16_t(const unsigned int& x)
        : data(float_to_bf16_raw(static_cast<float>(x)))
    {
    }

    // cast to float
    CK_TILE_HOST_DEVICE
    explicit constexpr operator float() const { return bf16_to_float_raw(data); }

    // cast to float
    CK_TILE_HOST_DEVICE
    explicit constexpr operator double() const { return bf16_to_double_raw(data); }

    // cast to int
    CK_TILE_HOST_DEVICE
    explicit constexpr operator int() const { return static_cast<int>(bf16_to_float_raw(data)); }

    // internal access
    CK_TILE_HOST_DEVICE
    constexpr raw_type& get() { return data; }

    CK_TILE_HOST_DEVICE
    constexpr raw_type get() const { return data; }
};

using bf16_t     = bfloat16_t;
using bf16_raw_t = typename bf16_t::raw_type;

// round to nearest
CK_TILE_HOST_DEVICE
uint16_t float_to_bf16_rtn_raw(float f)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {f};
    if(~u.int32 & 0x7f800000)
    {
        // When the exponent bits are not all 1s, then the value is zero, normal,
        // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
        // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
        // This causes the bfloat16's mantissa to be incremented by 1 if the 16
        // least significant bits of the float mantissa are greater than 0x8000,
        // or if they are equal to 0x8000 and the least significant bit of the
        // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
        // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
        // has the value 0x7f, then incrementing it causes it to become 0x00 and
        // the exponent is incremented by one, which is the next higher FP value
        // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
        // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
        // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
        // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
        // incrementing it causes it to become an exponent of 0xFF and a mantissa
        // of 0x00, which is Inf, the next higher value to the unrounded value.
        u.int32 += 0x7fff + ((u.int32 >> 16) & 1); // Round to nearest, round to even
    }
    else if(u.int32 & 0xffff)
    {
        // When all of the exponent bits are 1, the value is Inf or NaN.
        // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
        // mantissa bit. Quiet NaN is indicated by the most significant mantissa
        // bit being 1. Signaling NaN is indicated by the most significant
        // mantissa bit being 0 but some other bit(s) being 1. If any of the
        // lower 16 bits of the mantissa are 1, we set the least significant bit
        // of the bfloat16 mantissa, in order to preserve signaling NaN in case
        // the bloat16's mantissa bits are all 0.
        u.int32 |= 0x10000; // Preserve signaling NaN
    }
    return uint16_t(u.int32 >> 16);
}

// Truncate instead of rounding, preserving SNaN
CK_TILE_HOST_DEVICE
uint16_t float_to_bf16_truc_nan_raw(float f)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {f};
    return uint16_t(u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff));
}

// Fast truncate instead of rounding, RTZ
CK_TILE_HOST_DEVICE
uint16_t float_to_bf16_truc_raw(float f)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {f};
    return uint16_t(u.int32 >> 16);
}

template <bf16_rounding_mode rounding>
CK_TILE_HOST_DEVICE uint16_t float_to_bf16_raw(float f, constant<rounding>)
{
    if constexpr(rounding == bf16_rounding_mode::standard)
        return float_to_bf16_rtn_raw(f);
    else if constexpr(rounding == bf16_rounding_mode::truncate_with_nan)
        return float_to_bf16_truc_nan_raw(f);
    else
        return float_to_bf16_truc_raw(f);
}

template <bf16_rounding_mode rounding>
CK_TILE_HOST_DEVICE uint16_t double_to_bf16_raw(double f, constant<rounding>)
{
    return float_to_bf16_raw(static_cast<float>(f), constant<rounding>{});
}

CK_TILE_HOST_DEVICE
float bf16_to_float_raw(uint16_t x)
{
    union
    {
        uint32_t int32;
        float fp32;
    } u = {uint32_t(x) << 16};
    return u.fp32;
}

CK_TILE_HOST_DEVICE
double bf16_to_double_raw(uint16_t x) { return static_cast<double>(bf16_to_float_raw(x)); }

template <bf16_rounding_mode rounding =
              static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
CK_TILE_HOST_DEVICE bfloat16_t float_to_bf16(float f, constant<rounding>)
{
    return bfloat16_t::bit_cast(float_to_bf16_raw(f, constant<rounding>{}));
}

template <bf16_rounding_mode rounding =
              static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
CK_TILE_HOST_DEVICE bfloat16_t double_to_bf16(double f, constant<rounding>)
{
    return bfloat16_t::bit_cast(double_to_bf16_raw(f, constant<rounding>{}));
}

CK_TILE_HOST_DEVICE
float bf16_to_float(bfloat16_t x) { return static_cast<float>(x); }

CK_TILE_HOST_DEVICE
double bf16_to_double(bfloat16_t x) { return static_cast<double>(x); }

template <bf16_rounding_mode rounding =
              static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
CK_TILE_HOST_DEVICE bfloat16_t fp16_to_bf16(half_t f, constant<rounding> = {})
{
    return bfloat16_t::bit_cast(float_to_bf16_raw(static_cast<float>(f), constant<rounding>{}));
}

CK_TILE_HOST_DEVICE
half_t bf16_to_fp16(bfloat16_t x) { return float_to_fp16(static_cast<float>(x)); }

template <class T>
struct numeric_limits;

template <>
struct numeric_limits<bfloat16_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t min() { return bfloat16_t::bit_cast(0x0080); }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t lowest()
    {
        return bfloat16_t::bit_cast(0xff7f);
    }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t max() { return bfloat16_t::bit_cast(0x7f7f); }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t epsilon()
    {
        return bfloat16_t::bit_cast(0x1000);
    }

    // maximum rounding error
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t round_error() { return bfloat16_t(0.5f); }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t infinity()
    {
        return bfloat16_t::bit_cast(0x7f80);
    }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t quiet_NaN()
    {
        return bfloat16_t::bit_cast(0x7FFF);
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t signaling_NaN()
    {
        return bfloat16_t::bit_cast(0x7FFF);
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t denorm_min()
    {
        return bfloat16_t::bit_cast(0x0001);
    }
};

CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, bfloat16_t)

// math
CK_TILE_HOST_DEVICE
bfloat16_t abs(const bfloat16_t& x) { return bfloat16_t::bit_cast(x.get() & 0x7fff); }

CK_TILE_HOST_DEVICE
bool isnan(const bfloat16_t& x)
{
    uint16_t xx = x.get();
    return (xx & 0x7FFF) > 0x7C00;
}

CK_TILE_DEVICE
bfloat16_t sqrt(bfloat16_t x)
{
    return static_cast<bfloat16_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x)));
};

CK_TILE_DEVICE
bfloat16_t exp(bfloat16_t x) { return static_cast<bfloat16_t>(__expf(static_cast<float>(x))); };

CK_TILE_DEVICE
bfloat16_t exp2(bfloat16_t x) { return static_cast<bfloat16_t>(exp2f(static_cast<float>(x))); };

CK_TILE_DEVICE
bfloat16_t log(bfloat16_t x) { return static_cast<bfloat16_t>(__logf(static_cast<float>(x))); };

} // namespace ck_tile
