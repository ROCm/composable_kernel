// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"

// these conversions are disabled if native conversions available
#if !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
#if defined CK_ENABLE_FP8 || defined CK_ENABLE_BF8
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

template <typename X, typename Y, bool negative_zero_nan, bool clip, bool stoch>
__host__ __device__ Y run_cast_to_f8(X x, uint32_t rng)
{
    // fp8/bf8 exponent/mantissa layout
    constexpr int out_exp  = NumericUtils<Y>::exp;
    constexpr int out_mant = NumericUtils<Y>::mant;

    // original type exponent/mantissa layout
    constexpr int in_exp  = NumericUtils<X>::exp;
    constexpr int in_mant = NumericUtils<X>::mant;

    int exponent;
    uint32_t head, mantissa, sign;
    // nan code is same for float and half
    constexpr Y nan_code        = 0x80;
    constexpr uint32_t nan_mask = NumericUtils<X>::nan_mask;

    // convert to bitwise
    using T_bitwise     = typename NumericUtils<X>::bitwise_type;
    T_bitwise x_bitwise = *(reinterpret_cast<T_bitwise*>(&x));

    // unpack the input, depends on datatype
    head     = x_bitwise & NumericUtils<X>::head_mask;
    mantissa = x_bitwise & NumericUtils<X>::mant_mask;
    exponent = (head >> in_mant) & NumericUtils<X>::exp_mask;
    sign     = head >> (in_exp + in_mant);

    uint32_t signed_inf   = (sign << (in_exp + in_mant)) + (((1 << in_exp) - 1) << in_mant);
    uint32_t drop_mask    = (1 << (in_mant - out_mant)) - 1;
    constexpr int max_exp = (1 << out_exp) - (negative_zero_nan ? 1 : 2);
    constexpr int exp_low_cutoff =
        (1 << (in_exp - 1)) - (1 << (out_exp - 1)) + 1 - (negative_zero_nan ? 1 : 0);

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

    // if input is half and output is bf8
    if((NumericUtils<X>::mant == 10) && (NumericUtils<Y>::mant == 2) && negative_zero_nan &&
       exponent == 0)
    {
        exponent += 1;
        while(mantissa < (1 << in_mant))
        {
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa &= ~(1 << in_mant);
    }

    // check if x is 0.0
    if(x_bitwise == 0)
        return 0;

    exponent -= exp_low_cutoff - 1;
    if(exponent <= 0)
        drop_mask = (1 << (in_mant - out_mant + 1 - exponent)) - 1;
    mantissa += 1 << in_mant;
    // apply random number if needed
    mantissa += (stoch ? rng : mantissa) & drop_mask;
    if(mantissa >= (2 << in_mant))
    {
        mantissa >>= 1;
        exponent++;
    }
    mantissa >>= (in_mant - out_mant);

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
            mantissa = (1 << out_mant) - 1;
            exponent = max_exp;
        }
        else
        {
            return signed_inf;
        }
    }

    // check if x is 0.0 or -0.0
    if(exponent == 0 && mantissa == 0)
        return negative_zero_nan ? 0 : (sign << (out_exp + out_mant));
    mantissa &= (1 << out_mant) - 1;
    return (sign << (out_exp + out_mant)) | (exponent << out_mant) | mantissa;
}

template <typename X, typename Y, bool negative_zero_nan>
__host__ __device__ Y run_cast_from_f8(X x)
{
    // fp8/bf8 exponent/mantissa layout
    constexpr int in_exp  = NumericUtils<X>::exp;
    constexpr int in_mant = NumericUtils<X>::mant;

    // resulting type exponent/mantissa layout
    constexpr int out_exp  = NumericUtils<Y>::exp;
    constexpr int out_mant = NumericUtils<Y>::mant;

    // prepare the codes
    constexpr X nan_code = 0x80;
    Y Inf, NegInf, NaN, Neg0;
    using T_bitwise = typename NumericUtils<Y>::bitwise_type;

    constexpr T_bitwise Inf_bitwise    = NumericUtils<Y>::Inf;
    constexpr T_bitwise NegInf_bitwise = NumericUtils<Y>::NegInf;
    constexpr T_bitwise NaN_bitwise    = NumericUtils<Y>::NaN;
    constexpr T_bitwise Neg0_bitwise   = NumericUtils<Y>::Neg0;

    Inf    = *(reinterpret_cast<const Y*>(&Inf_bitwise));
    NegInf = *(reinterpret_cast<const Y*>(&NegInf_bitwise));
    NaN    = *(reinterpret_cast<const Y*>(&NaN_bitwise));
    Neg0   = *(reinterpret_cast<const Y*>(&Neg0_bitwise));

    // check if x is 0.0
    if(x == 0)
        return static_cast<Y>(0);

    // unpack the input
    uint32_t sign     = x >> (in_exp + in_mant);
    uint32_t mantissa = x & ((1 << in_mant) - 1);
    int exponent      = (x & 0x7F) >> in_mant;

    constexpr int exp_low_cutoff =
        (1 << (out_exp - 1)) - (1 << (in_exp - 1)) + 1 - (negative_zero_nan ? 1 : 0);
    T_bitwise retval;

    if constexpr(negative_zero_nan)
    {
        if(x == nan_code)
            return NaN;
    }
    else
    {
        if(x == nan_code)
            return Neg0;
        if(exponent == ((1 << in_exp) - 1))
            return (mantissa == 0) ? (sign ? NegInf : Inf) : NaN;
    }

    if((NumericUtils<Y>::mant == 10) && (NumericUtils<X>::mant == 2) && !negative_zero_nan)
    {
        retval = x;
        retval <<= 8;
        return *(reinterpret_cast<const Y*>(&retval));
    }

    // subnormal input
    if(exponent == 0)
    {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        exponent++;
        while(mantissa < (1 << in_mant))
        {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= ((1 << in_mant) - 1);
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= out_mant - in_mant;

    // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << out_mant;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    retval = (sign << (out_exp + out_mant)) | (exponent << out_mant) | mantissa;
    return *(reinterpret_cast<const Y*>(&retval));
}

} // namespace

template <typename X, typename Y, bool negative_zero_nan, bool clip, bool stoch>
__host__ __device__ Y cast_to_f8(X x, uint32_t rng)
{
    // check datatypes
    constexpr bool is_half  = std::is_same<X, half_t>::value;
    constexpr bool is_float = std::is_same<X, float>::value;
    static_assert(is_half || is_float, "Only half and float can be casted.");

    return run_cast_to_f8<X, Y, negative_zero_nan, clip, stoch>(x, rng);
}

template <typename X, typename Y, bool negative_zero_nan>
__host__ __device__ Y cast_from_f8(X x)
{
    // check datatype
    constexpr bool is_half  = std::is_same<Y, half_t>::value;
    constexpr bool is_float = std::is_same<Y, float>::value;
    static_assert(is_half || is_float, "only half and float are supported.");

    return run_cast_from_f8<X, Y, negative_zero_nan>(x);
}

} // namespace ck::utils
#endif // #if defined CK_ENABLE_FP8 || defined CK_ENABLE_BF8
#endif // #if !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
