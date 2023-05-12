// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/statically_indexed_array.hpp"

namespace ck {

using f8_t = uint8_t;

// fp8 rounding modes
enum class f8_rounding_mode
{
    standard,
    stochastic
};

// cast fp32 to fp8
template <bool negative_zero_nan, bool clip, bool stoch>
__host__ __device__ f8_t cast_to_f8(float x, uint32_t rng)
{
    // fp8 exponent/mantissa layout
    constexpr int we_f8 = 4;
    constexpr int wm_f8 = 3;

    // fp32 exponent/mantissa layout
    // constexpr int we_f32 = 8;
    constexpr int wm_f32 = 23;

    uint32_t x_bitwise;
    x_bitwise = *(reinterpret_cast<uint32_t*>(&x));

    // unpack the input
    uint32_t head, mantissa;
    int exponent;
    uint32_t sign;

    head     = x_bitwise & 0xFF800000;
    mantissa = x_bitwise & 0x7FFFFF;
    exponent = (head >> 23) & 0xFF;
    sign     = head >> 31;

    uint32_t signed_inf = (sign << (we_f8 + wm_f8)) + (((1 << we_f8) - 1) << wm_f8);

    if(negative_zero_nan)
    {
        if((x_bitwise & 0x7F800000) == 0x7F800000)
            return 0x80;
    }
    else
    {
        if((x_bitwise & 0x7F800000) == 0x7F800000)
            return signed_inf + (mantissa != 0 ? 1 : 0);
    }

    if(x_bitwise == 0)
        return 0;

    uint32_t drop_mask       = (1 << (wm_f32 - wm_f8)) - 1;
    const int max_exp        = (1 << we_f8) - (negative_zero_nan ? 1 : 2);
    const int exp_low_cutoff = 0x80 - (1 << (we_f8 - 1)) + 1 - (negative_zero_nan ? 1 : 0);

    exponent -= exp_low_cutoff - 1;
    if(exponent <= 0)
        drop_mask = (1 << (wm_f32 - wm_f8 + 1 - exponent)) - 1;
    mantissa += 1 << wm_f32;
    mantissa += (stoch ? rng : mantissa) & drop_mask;
    if(mantissa >= (2 << wm_f32))
    {
        mantissa >>= 1;
        exponent++;
    }
    mantissa >>= (wm_f32 - wm_f8);

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
            mantissa = (1 << wm_f8) - 1;
            exponent = max_exp;
        }
        else
        {
            return signed_inf;
        }
    }
    if(exponent == 0 && mantissa == 0)
        return negative_zero_nan ? 0 : (sign << 7);
    mantissa &= (1 << wm_f8) - 1;
    return (sign << 7) | (exponent << wm_f8) | mantissa;
}

// cast fp8 to fp32
template <bool negative_zero_nan>
__host__ __device__ float cast_from_f8(f8_t x)
{
    // fp8 exponent/mantissa layout
    constexpr int we_f8 = 4;
    constexpr int wm_f8 = 3;

    // fp32 exponent/mantissa layout
    constexpr int we_f32 = 8;
    constexpr int wm_f32 = 23;

    float fInf, fNegInf, fNaN, fNeg0;
    const uint32_t ifInf    = 0x7F800000;
    const uint32_t ifNegInf = 0xFF800000;
    const uint32_t ifNaN    = 0x7F800001;
    const uint32_t ifNeg0   = 0x80000000;
    fInf                    = *(reinterpret_cast<const float*>(&ifInf));
    fNegInf                 = *(reinterpret_cast<const float*>(&ifNegInf));
    fNaN                    = *(reinterpret_cast<const float*>(&ifNaN));
    fNeg0                   = *(reinterpret_cast<const float*>(&ifNeg0));

    if(x == 0)
        return static_cast<float>(0);

    // unpack the input
    uint32_t sign     = x >> (we_f8 + wm_f8);
    uint32_t mantissa = x & ((1 << wm_f8) - 1);
    int exponent      = (x & 0x7F) >> wm_f8;

    if(negative_zero_nan)
    {
        if(x == 0x80)
            return fNaN;
    }
    else
    {
        if(x == 0x80)
            return fNeg0;
        if(exponent == ((1 << we_f8) - 1))
            return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
    }

    uint32_t retval;
    const int exp_low_cutoff =
        (1 << (we_f32 - 1)) - (1 << (we_f8 - 1)) + 1 - (negative_zero_nan ? 1 : 0);

    // subnormal input
    if(exponent == 0)
    {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        int sh = 1 + __builtin_clz(mantissa) - ((1 + we_f32 + wm_f32) - wm_f8);
        mantissa <<= sh;
        exponent += 1 - sh;
        /*
        exponent++;
        while(mantissa<(1<<wm)) {
        mantissa <<= 1;
        exponent--;
        }
        */
        mantissa &= ((1 << wm_f8) - 1);
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= wm_f32 - wm_f8;

    // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << wm_f32;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    retval = (sign << (we_f32 + wm_f32)) | (exponent << wm_f32) | mantissa;
    return *(reinterpret_cast<const float*>(&retval));
}

} // namespace ck
