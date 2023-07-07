// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"

namespace ck {

// fp8 rounding modes
// use standard for rounding to nearest, the faster one
// use stochastic for stochastic rounding, helps to avoid error accumulation
enum class f8_rounding_mode
{
    standard,
    stochastic
};

} // namespace ck

namespace ck::utils {

namespace {

template <typename T, bool negative_zero_nan, bool clip, bool stoch>
__host__ __device__ f8_t run_cast_to_f8(T x, uint32_t rng)
{
    // check data type
    constexpr bool is_half  = std::is_same<T, half_t>::value;
    constexpr bool is_float = std::is_same<T, float>::value;

    // fp8 exponent/mantissa layout
    constexpr int f8_exp  = 4;
    constexpr int f8_mant = 3;

    // resulting type exponent/mantissa layout
    constexpr int type_exp  = is_half ? 5 : 8;
    constexpr int type_mant = is_half ? 10 : 23;

    int exponent;
    uint32_t head, mantissa, sign;
    // nan code is same for float and half
    constexpr uint8_t nan_code  = 0x80;
    constexpr uint32_t nan_mask = is_half ? 0x7C00 : 0x7F800000;

    // convert to bitwise
    typedef typename std::conditional<std::is_same<T, half_t>::value, uint16_t, uint32_t>::type
        T_bitwise;
    T_bitwise x_bitwise = *(reinterpret_cast<T_bitwise*>(&x));

    // unpack the input, depends on datatype
    if constexpr(is_float)
    {
        head     = x_bitwise & 0xFF800000;
        mantissa = x_bitwise & 0x7FFFFF;
        exponent = (head >> type_mant) & 0xFF;
        sign     = head >> (type_exp + type_mant);
    }
    else if constexpr(is_half)
    {
        head     = x_bitwise & 0xFC00;
        mantissa = x_bitwise & 0x3FF;
        exponent = (head >> type_mant) & 0x1F;
        sign     = head >> (type_exp + type_mant);
    }

    uint32_t signed_inf   = (sign << (type_exp + type_mant)) + (((1 << type_exp) - 1) << type_mant);
    uint32_t drop_mask    = (1 << (type_mant - f8_mant)) - 1;
    constexpr int max_exp = (1 << f8_exp) - (negative_zero_nan ? 1 : 2);
    constexpr int exp_low_cutoff =
        (1 << (type_exp - 1)) - (1 << (f8_exp - 1)) + 1 - (negative_zero_nan ? 1 : 0);

    if constexpr(negative_zero_nan)
    {
        if((x_bitwise & nan_mask) == nan_mask)
            return nan_code;
    }
    else
    {
        if((x_bitwise & nan_mask) == nan_mask)
            return signed_inf + (mantissa != 0 ? 1 : 0);
    }

    // check if x is 0.0
    if(x_bitwise == 0)
        return 0;

    exponent -= exp_low_cutoff - 1;
    if(exponent <= 0)
        drop_mask = (1 << (type_mant - f8_mant + 1 - exponent)) - 1;
    mantissa += 1 << type_mant;
    // apply random number if needed
    mantissa += (stoch ? rng : mantissa) & drop_mask;
    if(mantissa >= (2 << type_mant))
    {
        mantissa >>= 1;
        exponent++;
    }
    mantissa >>= (type_mant - f8_mant);

    // check negative exponent
    if(exponent <= 0)
    {
        if(x_bitwise == 0)
            return 0;
        else
        {
            // subnormal range; represented by a subnormal float8 (exponent 0)
            // and involves loss of accuracy
            mantissa >>= 1 - exponent;
            exponent = 0;
        }
    }
    // above range: quantize to maximum possible float of the same sign
    else if(exponent > max_exp)
    {
        if(clip)
        {
            mantissa = (1 << f8_mant) - 1;
            exponent = max_exp;
        }
        else
        {
            return signed_inf;
        }
    }

    // check if x is 0.0 or -0.0
    if(exponent == 0 && mantissa == 0)
        return negative_zero_nan ? 0 : (sign << (f8_exp + f8_mant));
    mantissa &= (1 << f8_mant) - 1;
    return (sign << (f8_exp + f8_mant)) | (exponent << f8_mant) | mantissa;
}

template <typename T, bool negative_zero_nan>
__host__ __device__ T run_cast_from_f8(f8_t x)
{
    // check data type
    constexpr bool is_half  = std::is_same<T, half_t>::value;
    constexpr bool is_float = std::is_same<T, float>::value;

    // fp8 exponent/mantissa layout
    constexpr int f8_exp  = 4;
    constexpr int f8_mant = 3;

    // resulting type exponent/mantissa layout
    constexpr int type_exp  = is_half ? 5 : 8;
    constexpr int type_mant = is_half ? 10 : 23;

    // prepare the codes
    constexpr uint8_t nan_code = 0x80;
    T fInf, fNegInf, fNaN, fNeg0;
    if constexpr(is_half)
    {
        constexpr uint16_t ihInf    = 0x7C00;
        constexpr uint16_t ihNegInf = 0xFC00;
        constexpr uint16_t ihNaN    = 0x7C01;
        constexpr uint16_t ihNeg0   = 0x8000;
        fInf                        = *(reinterpret_cast<const half_t*>(&ihInf));
        fNegInf                     = *(reinterpret_cast<const half_t*>(&ihNegInf));
        fNaN                        = *(reinterpret_cast<const half_t*>(&ihNaN));
        fNeg0                       = *(reinterpret_cast<const half_t*>(&ihNeg0));
    }
    else if constexpr(is_float)
    {
        constexpr uint32_t ifInf    = 0x7F800000;
        constexpr uint32_t ifNegInf = 0xFF800000;
        constexpr uint32_t ifNaN    = 0x7F800001;
        constexpr uint32_t ifNeg0   = 0x80000000;
        fInf                        = *(reinterpret_cast<const float*>(&ifInf));
        fNegInf                     = *(reinterpret_cast<const float*>(&ifNegInf));
        fNaN                        = *(reinterpret_cast<const float*>(&ifNaN));
        fNeg0                       = *(reinterpret_cast<const float*>(&ifNeg0));
    }

    // unpack the input
    uint32_t sign     = x >> (f8_exp + f8_mant);
    uint32_t mantissa = x & ((1 << f8_mant) - 1);
    int exponent      = (x & 0x7F) >> f8_mant;

    constexpr int exp_low_cutoff =
        (1 << (type_exp - 1)) - (1 << (f8_exp - 1)) + 1 - (negative_zero_nan ? 1 : 0);
    typename std::conditional<std::is_same<T, half_t>::value, uint16_t, uint32_t>::type retval;

    if constexpr(negative_zero_nan)
    {
        if(x == nan_code)
            return fNaN;
    }
    else
    {
        if(x == nan_code)
            return fNeg0;
        if(exponent == ((1 << f8_exp) - 1))
            return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
    }

    // subnormal input
    if(exponent == 0)
    {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        int sh = 1 + __builtin_clz(mantissa) - ((1 + type_exp + type_mant) - f8_mant);
        mantissa <<= sh;
        mantissa &= ((1 << f8_mant) - 1);
        exponent += 1 - sh;
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= type_mant - f8_mant;

    // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << type_mant;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    retval = (sign << (type_exp + type_mant)) | (exponent << type_mant) | mantissa;
    return *(reinterpret_cast<const T*>(&retval));
}

} // namespace

template <typename T, bool negative_zero_nan, bool clip, bool stoch>
__host__ __device__ f8_t cast_to_f8(T x, uint32_t rng)
{
    // check datatype
    constexpr bool is_half  = std::is_same<T, half_t>::value;
    constexpr bool is_float = std::is_same<T, float>::value;
    static_assert(is_half || is_float, "Only half and float can be casted to f8.");

    return run_cast_to_f8<T, negative_zero_nan, clip, stoch>(x, rng);
}

template <typename T, bool negative_zero_nan>
__host__ __device__ T cast_from_f8(f8_t x)
{
    // check datatype
    constexpr bool is_half  = std::is_same<T, half_t>::value;
    constexpr bool is_float = std::is_same<T, float>::value;
    static_assert(is_half || is_float, "only half and float are supported.");

    // check if x is 0.0
    if(x == 0)
        return static_cast<T>(0);

    return run_cast_from_f8<T, negative_zero_nan>(x);
}

} // namespace ck::utils
