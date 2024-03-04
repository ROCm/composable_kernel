// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/limits.hpp"
#include "ck_tile/core/utility/random.hpp"
#include "ck_tile/core/numeric/arithmetic.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/limits.hpp"
#include <stdint.h>
#include <type_traits>

#pragma once

namespace ck_tile {

// fp8 rounding modes
// use standard for rounding to nearest, the faster one
// use stochastic for stochastic rounding, helps to avoid error accumulation
enum class fp8_rounding_mode
{
    standard = 0,
    stochastic
};

/*
 *                ______________NANOO_________________    |   ______________IEEE________________
 *                   e4m3               e5m2              |    e4m3                e5m2
 *      bias :        8                  16               |     7                   15
 *      inf  :  1.0000.000           1.00000.00           |    N/A              s.11111.00
 *      Nan  :  1.0000.000           1.00000.00           | s.1111.111          s.11111.{01, 10, 11}
 *      zero :  0.0000.000           0.00000.00           | s.0000.000          s.00000.00
 * Max(norm) :  s.1111.111 (240)     s.11111.11(57344)    | s.1111.110(448)     s.11110.11(57344)
 * Max(snorm):  s.0000.111           s.00000.11           | s.0000.111(448)     s.00000.11(57344)
 *                0.0068359375         2.288818e-05       |   0.013671875         4.57763671875e-05
 * Min(norm) :  s.0001.000           s.00001.00           | s.0001.000          s.00001.00
 *                2^-7(0.00078125)     2^-15(3.05176e-05) |   2^-6(0.015625)      2^-14(6.10352e-05)
 * Min(snorm):  s.0000.001           s.00000.01           | s.0000.001          s.00000.01
 *                2^-10(0.00097656)    2^-17(7.629395e-06)|   2^-9(0.001953125)   2^-16(1.52588e-05)
 */

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE uint8_t float_to_fp8_raw(float, constant<rounding> = {});

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE uint8_t float_to_bf8_raw(float, constant<rounding> = {});

CK_TILE_HOST_DEVICE float fp8_to_float_raw(uint8_t);
CK_TILE_HOST_DEVICE float bf8_to_float_raw(uint8_t);

struct alignas(1) float8_e4m3_t
{
    static constexpr int exponent = 4;
    static constexpr int mantissa = 3;
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    static constexpr int bias = 1 << (exponent - 1); // NANOO
#else
    static constexpr int bias        = (1 << (exponent - 1)) - 1; // IEEE
#endif
    using raw_type = uint8_t;
    raw_type data;

    CK_TILE_HOST_DEVICE
    static constexpr float8_e4m3_t bit_cast(raw_type x)
    {
        float8_e4m3_t y;
        y.data = x;
        return y;
    }

    // constructor
    constexpr float8_e4m3_t() : data() {}

    // construct from float
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e4m3_t(const float& x) : data(float_to_fp8_raw(x)) {}

    // construct from int
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e4m3_t(const int& x) : data(float_to_fp8_raw(static_cast<float>(x)))
    {
    }

    // construct from unsigned int
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e4m3_t(const unsigned int& x)
        : data(float_to_fp8_raw(static_cast<float>(x)))
    {
    }

    // cast to float
    CK_TILE_HOST_DEVICE
    explicit constexpr operator float() const { return fp8_to_float_raw(data); }

    // cast to int
    CK_TILE_HOST_DEVICE
    explicit constexpr operator int() const { return static_cast<int>(fp8_to_float_raw(data)); }

    // internal access
    CK_TILE_HOST_DEVICE
    constexpr raw_type& get() { return data; }

    CK_TILE_HOST_DEVICE
    constexpr raw_type get() const { return data; }
};
using fp8_t     = float8_e4m3_t;
using fp8_raw_t = typename fp8_t::raw_type;

struct alignas(1) float8_e5m2_t
{
    static constexpr int exponent = 5;
    static constexpr int mantissa = 2;
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    static constexpr int bias = 1 << (exponent - 1); // NANOO
#else
    static constexpr int bias        = (1 << (exponent - 1)) - 1; // IEEE
#endif
    using raw_type = uint8_t;
    raw_type data;

    CK_TILE_HOST_DEVICE
    static constexpr float8_e5m2_t bit_cast(raw_type x)
    {
        float8_e5m2_t y;
        y.data = x;
        return y;
    }

    // constructor
    constexpr float8_e5m2_t() : data() {}

    // construct from float
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e5m2_t(const float& x) : data(float_to_bf8_raw(x)) {}

    // construct from int
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e5m2_t(const int& x) : data(float_to_bf8_raw(static_cast<float>(x)))
    {
    }

    // construct from unsigned int
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e5m2_t(const unsigned int& x)
        : data(float_to_bf8_raw(static_cast<float>(x)))
    {
    }

    // cast to float
    CK_TILE_HOST_DEVICE
    explicit constexpr operator float() const { return bf8_to_float_raw(data); }

    // cast to int
    CK_TILE_HOST_DEVICE
    explicit constexpr operator int() const { return static_cast<int>(bf8_to_float_raw(data)); }

    // internal access
    CK_TILE_HOST_DEVICE
    constexpr raw_type& get() { return data; }

    CK_TILE_HOST_DEVICE
    constexpr raw_type get() const { return data; }
};
using bf8_t     = float8_e5m2_t;
using bf8_raw_t = typename bf8_t::raw_type;

// below is sw fp8 conversion, not utilizing hw instruction
namespace impl {

template <typename X, typename Y, bool negative_zero_nan, bool clip, bool stoch>
CK_TILE_HOST_DEVICE Y run_cast_to_f8(X x, uint32_t rng)
{
    // fp8/bf8 exponent/mantissa layout
    constexpr int out_exp  = numeric_utils<Y>::exp;
    constexpr int out_mant = numeric_utils<Y>::mant;

    // original type exponent/mantissa layout
    constexpr int in_exp  = numeric_utils<X>::exp;
    constexpr int in_mant = numeric_utils<X>::mant;

    int exponent, bias;
    uint32_t head, mantissa, sign;
    // nan code is same for float and half
    constexpr Y nan_code        = __builtin_bit_cast(Y, static_cast<uint8_t>(0x80));
    constexpr uint32_t nan_mask = numeric_utils<X>::nan_mask;

    // convert to bitwise
    using T_bitwise     = typename numeric_utils<X>::bitwise_type;
    T_bitwise x_bitwise = *(reinterpret_cast<T_bitwise*>(&x));

    // unpack the input, depends on datatype
    head     = x_bitwise & numeric_utils<X>::head_mask;
    mantissa = x_bitwise & numeric_utils<X>::mant_mask;
    exponent = (head >> in_mant) & numeric_utils<X>::exp_mask;
    sign     = head >> (in_exp + in_mant);
    bias     = numeric_utils<X>::bias;

    uint32_t signed_inf   = (sign << (in_exp + in_mant)) + (((1 << in_exp) - 1) << in_mant);
    uint32_t drop_mask    = (1 << (in_mant - out_mant)) - 1;
    constexpr int max_exp = (1 << out_exp) - (negative_zero_nan ? 1 : 2);

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
        return __builtin_bit_cast(Y, static_cast<uint8_t>(0));

    // First need to check if it is normal or denorm as there is a difference of implict 1
    // Then need to adjust the exponent to align with the F8 exponent, in the meanwhile, shift
    // The mantissa. Then for stochastic rounding, add rng to mantissa and truncate. And for
    // RNE, no need to add rng. Then probably need to check whether there is carry and adjust
    // exponent and mantissa again3

    // For IEEE bias mode, the bias is 2^(k-1)-1 where k is the width of exponent bits
    const int out_bias                  = (1 << (out_exp - 1)) - 1 + (negative_zero_nan ? 1 : 0);
    const int out_denormal_act_exponent = 1 - out_bias; // actual exponent of f8 denormal
    // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
    // out_exponent is the converted f8 exponent with bias encoding
    // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
    // the difference needs to be adjusted and mantissa shifted
    int act_exponent, out_exponent, exponent_diff;

    if(exponent == 0)
    { // fp32/fp16 is in denormal.
        /* fp32 denormal is below 2^-127 so it is usually not a concern here, we mostly concern fp16
here. In this case, f8 is usually in denormal. But there could be exceptions. fp16 denormal has
exponent bias 15 while bf8 with NANOO has exponent bias 16. It means that there are some numbers in
fp16 denormal but they are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15. fp16 numbers
where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8 (NANOO) normal.
In this case, the fp16 mantissa should be shift left by 1 */
        act_exponent  = exponent - bias + 1;
        exponent_diff = out_denormal_act_exponent -
                        act_exponent; // actual exponent is exponent-bias+1 as it is denormal
    }
    else
    { // fp32/fp16 is normal with implicit 1
        act_exponent = exponent - bias;
        if(act_exponent <= out_denormal_act_exponent)
        {
            /* This is the case where fp32/fp16 is normal but it is in f8 denormal range.
   For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
   actual exponent is -7, it is actually larger due to the implict 1,
   Therefore it needs to be adjust to -6 and mantissa shift right by 1.
   So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
            exponent_diff = out_denormal_act_exponent - act_exponent;
        }
        else
        { // both fp32/fp16 and f8 are in normal range
            exponent_diff =
                0; // exponent_diff=0 does not mean there is no difference for this case,
            // act_exponent could be larger. Just that it does not need shift mantissa
        }
        mantissa += (1 << in_mant); // Add the implicit 1 into mantissa
    }

    bool midpoint = (mantissa & ((1 << (in_mant - out_mant + exponent_diff)) - 1)) ==
                    (1 << (in_mant - out_mant + exponent_diff - 1));
    /* This part is a bit tricky. The judgment of whether it is a tie needs to be done before we
 shift right as shift right could rip off some residual part and make something not midpoint look
 like midpoint. For example, the fp16 number 0x1002 (0 00100 0000000010), it is larger than
 midpoint, but after shift right by 4 bits, it would look like midpoint. */

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1 << in_mant);
    // if there is no implict 1, it  means the f8 is denormal and need to adjust to denorm exponent
    out_exponent =
        (act_exponent + exponent_diff) /*actual f8 exponent*/ + out_bias - (implicit_one ? 0 : 1);

    // Now we have the exponent and mantissa adjusted
    bool odd =
        mantissa &
        (1 << (in_mant - out_mant)); // if the least significant bit that is not truncated is 1
    mantissa += (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa)) & drop_mask;

    // Now we deal with overflow
    if(out_exponent == 0)
    {
        if((1 << in_mant) & mantissa)
        {
            out_exponent = 1; // denormal overflow to become normal, promote exponent
            // No need to make 1 implicit now as it will be addressed later
        }
    }
    else
    {
        if((1 << (in_mant + 1)) & mantissa)
        {
            mantissa >>= 1;
            out_exponent++;
            // No need to make 1 implicit now as it will be addressed later
        }
    }

    mantissa >>= (in_mant - out_mant);

    if(out_exponent > max_exp)
    {
        if(clip)
        {
            mantissa     = (1 << out_mant) - 1;
            out_exponent = max_exp;
        }
        else
        {
            return __builtin_bit_cast(Y, static_cast<uint8_t>(signed_inf));
        }
    }

    // check if x is 0.0 or -0.0
    if(out_exponent == 0 && mantissa == 0)
        return __builtin_bit_cast(
            Y, static_cast<uint8_t>(negative_zero_nan ? 0 : (sign << (out_exp + out_mant))));
    mantissa &= (1 << out_mant) - 1;
    return __builtin_bit_cast(Y,
                              static_cast<uint8_t>((sign << (out_exp + out_mant)) |
                                                   (out_exponent << out_mant) | mantissa));
}

template <typename X, typename Y, bool negative_zero_nan>
CK_TILE_HOST_DEVICE Y run_cast_from_f8(X x)
{
    // fp8/bf8 exponent/mantissa layout
    constexpr int in_exp  = numeric_utils<X>::exp;
    constexpr int in_mant = numeric_utils<X>::mant;

    // resulting type exponent/mantissa layout
    constexpr int out_exp  = numeric_utils<Y>::exp;
    constexpr int out_mant = numeric_utils<Y>::mant;
    uint8_t x_raw          = __builtin_bit_cast(uint8_t, x);

    // prepare the codes
    constexpr uint8_t nan_code = 0x80;
    Y Inf, NegInf, NaN, Neg0;
    using T_bitwise = typename numeric_utils<Y>::bitwise_type;

    constexpr T_bitwise Inf_bitwise    = numeric_utils<Y>::Inf;
    constexpr T_bitwise NegInf_bitwise = numeric_utils<Y>::NegInf;
    constexpr T_bitwise NaN_bitwise    = numeric_utils<Y>::NaN;
    constexpr T_bitwise Neg0_bitwise   = numeric_utils<Y>::Neg0;

    Inf    = *(reinterpret_cast<const Y*>(&Inf_bitwise));
    NegInf = *(reinterpret_cast<const Y*>(&NegInf_bitwise));
    NaN    = *(reinterpret_cast<const Y*>(&NaN_bitwise));
    Neg0   = *(reinterpret_cast<const Y*>(&Neg0_bitwise));

    // check if x is 0.0
    if(x_raw == 0)
        return static_cast<Y>(0);

    // unpack the input
    uint32_t sign     = x_raw >> (in_exp + in_mant);
    uint32_t mantissa = x_raw & ((1 << in_mant) - 1);
    int exponent      = (x_raw & 0x7F) >> in_mant;

    constexpr int exp_low_cutoff =
        (1 << (out_exp - 1)) - (1 << (in_exp - 1)) + 1 - (negative_zero_nan ? 1 : 0);
    T_bitwise retval;

    if constexpr(negative_zero_nan)
    {
        if(x_raw == nan_code)
            return NaN;
    }
    else
    {
        if(x_raw == nan_code)
            return Neg0;
        if(exponent == ((1 << in_exp) - 1))
            return (mantissa == 0) ? (sign ? NegInf : Inf) : NaN;
    }

    if((numeric_utils<Y>::mant == 10) && (numeric_utils<X>::mant == 2) && !negative_zero_nan)
    {
        retval = x_raw;
        retval <<= 8;
        return *(reinterpret_cast<const Y*>(&retval));
    }

    // subnormal input
    if(exponent == 0)
    {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        int sh = 1 + clz(mantissa) - (32 - in_mant);
        mantissa <<= sh;
        exponent += 1 - sh;
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

template <typename X, typename Y, bool negative_zero_nan, bool clip, bool stoch>
CK_TILE_HOST_DEVICE Y cast_to_f8(X x, uint32_t rng)
{
    // check datatypes
    constexpr bool is_half  = std::is_same<X, half_t>::value;
    constexpr bool is_float = std::is_same<X, float>::value;
    static_assert(is_half || is_float, "Only half and float can be casted.");

    return run_cast_to_f8<X, Y, negative_zero_nan, clip, stoch>(x, rng);
}

template <typename X, typename Y, bool negative_zero_nan>
CK_TILE_HOST_DEVICE Y cast_from_f8(X x)
{
    // check datatype
    constexpr bool is_half  = std::is_same<Y, half_t>::value;
    constexpr bool is_float = std::is_same<Y, float>::value;
    static_assert(is_half || is_float, "only half and float are supported.");

    return run_cast_from_f8<X, Y, negative_zero_nan>(x);
}
} // namespace impl

CK_TILE_HOST_DEVICE fp8_raw_t float_to_fp8_sr_raw(float x)
{
    constexpr int seed = 42;
    uint32_t rng       = prand_generator_t<float, seed>{}(reinterpret_cast<uintptr_t>(&x), x);
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    float max_fp8 = 240.0f;
    x             = x > max_fp8 ? max_fp8 : (x < -max_fp8 ? -max_fp8 : x);
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval      = x;
    uint32_t ival = 0;
    ival          = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
    val.i32val    = ival;
    return val.i8val[0]; // little endian
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr fp8_rounding_mode rm   = fp8_rounding_mode::stochastic;
    return bit_cast<fp8_raw_t>(impl::cast_to_f8<float,
                                                fp8_t,
                                                negative_zero_nan,
                                                clip,
                                                (rm == fp8_rounding_mode::stochastic)>(x, rng));
#endif
}

CK_TILE_HOST_DEVICE bf8_raw_t float_to_bf8_sr_raw(float x)
{
    constexpr int seed = 42;
    uint32_t rng       = prand_generator_t<float, seed>{}(reinterpret_cast<uintptr_t>(&x), x);
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval      = x;
    uint32_t ival = 0;
    ival          = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
    val.i32val    = ival;
    return val.i8val[0]; // little endian
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr fp8_rounding_mode rm   = fp8_rounding_mode::stochastic;
    return bit_cast<bf8_raw_t>(impl::cast_to_f8<float,
                                                bf8_t,
                                                negative_zero_nan,
                                                clip,
                                                (rm == fp8_rounding_mode::stochastic)>(x, rng));
#endif
}

CK_TILE_HOST_DEVICE fp8_raw_t float_to_fp8_rtn_raw(float x)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    float max_fp8 = 240.0f;
    x             = x > max_fp8 ? max_fp8 : (x < -max_fp8 ? -max_fp8 : x);
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval      = x;
    uint32_t ival = 0;
    ival       = __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival, false); // false -> WORD0
    val.i32val = ival;
    return val.i8val[0];
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr fp8_rounding_mode rm   = fp8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return bit_cast<fp8_raw_t>(impl::cast_to_f8<float,
                                                fp8_t,
                                                negative_zero_nan,
                                                clip,
                                                (rm == fp8_rounding_mode::stochastic)>(x, rng));
#endif
}
CK_TILE_HOST_DEVICE bf8_raw_t float_to_bf8_rtn_raw(float x)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval      = x;
    uint32_t ival = 0;
    ival       = __builtin_amdgcn_cvt_pk_bf8_f32(val.fval, val.fval, ival, false); // false -> WORD0
    val.i32val = ival;
    return val.i8val[0];
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr fp8_rounding_mode rm   = fp8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return bit_cast<bf8_raw_t>(impl::cast_to_f8<float,
                                                bf8_t,
                                                negative_zero_nan,
                                                clip,
                                                (rm == fp8_rounding_mode::stochastic)>(x, rng));
#endif
}

// clang-format off
template<fp8_rounding_mode rounding>
CK_TILE_HOST_DEVICE fp8_raw_t float_to_fp8_raw(float x, constant<rounding>)
{
    if      constexpr (rounding == fp8_rounding_mode::standard)   return float_to_fp8_rtn_raw(x);
    else if constexpr (rounding == fp8_rounding_mode::stochastic) return float_to_fp8_sr_raw(x);
    else return fp8_raw_t{0};
}

template<fp8_rounding_mode rounding>
CK_TILE_HOST_DEVICE bf8_raw_t float_to_bf8_raw(float x, constant<rounding>)
{
    if      constexpr (rounding == fp8_rounding_mode::standard)   return float_to_bf8_rtn_raw(x);
    else if constexpr (rounding == fp8_rounding_mode::stochastic) return float_to_bf8_sr_raw(x);
    else return bf8_raw_t{0};
}

CK_TILE_HOST_DEVICE float fp8_to_float_raw(fp8_raw_t x)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);
    fval            = __builtin_amdgcn_cvt_f32_fp8(i32val, 0);
    // asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    constexpr bool negative_zero_nan = true;
    return impl::cast_from_f8<fp8_t, float, negative_zero_nan>(fp8_t::bit_cast(x));
#endif
}

CK_TILE_HOST_DEVICE float bf8_to_float_raw(bf8_raw_t x)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);
    fval            = __builtin_amdgcn_cvt_f32_bf8(i32val, 0);
    // asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    constexpr bool negative_zero_nan = true;
    return impl::cast_from_f8<bf8_t, float, negative_zero_nan>(bf8_t::bit_cast(x));
#endif
}

template<fp8_rounding_mode rounding>
CK_TILE_HOST_DEVICE float8_e4m3_t float_to_fp8(float x, constant<rounding>)
{
    return float8_e4m3_t::bit_cast(float_to_fp8_raw(x, constant<rounding>{}));
}

template<fp8_rounding_mode rounding>
CK_TILE_HOST_DEVICE float8_e5m2_t float_to_bf8(float x, constant<rounding>)
{
    return float8_e5m2_t::bit_cast(float_to_bf8_raw(x, constant<rounding>{}));
}

CK_TILE_HOST_DEVICE float fp8_to_float(float8_e4m3_t x)
{
    return fp8_to_float_raw(x.get());
}

CK_TILE_HOST_DEVICE float bf8_to_float(float8_e5m2_t x)
{
    return bf8_to_float_raw(x.get());
}

// clang-format on

template <typename T>
struct numeric_utils;

template <>
struct numeric_utils<fp8_t>
{
    static constexpr int exp  = fp8_t::exponent;
    static constexpr int mant = fp8_t::mantissa;
    static constexpr int bias = fp8_t::bias;
};

template <>
struct numeric_utils<bf8_t>
{
    static constexpr int exp  = bf8_t::exponent;
    static constexpr int mant = bf8_t::mantissa;
    static constexpr int bias = bf8_t::bias;
};

template <class T>
struct numeric_limits;

template <>
struct numeric_limits<fp8_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr fp8_t min() { return fp8_t::bit_cast(0x08); }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr fp8_t lowest() { return fp8_t::bit_cast(0xff); }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr fp8_t max() { return fp8_t::bit_cast(0x7f); }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr fp8_t epsilon() { return fp8_t::bit_cast(0x20); }

    // maximum rounding error
    CK_TILE_HOST_DEVICE static constexpr fp8_t round_error() { return fp8_t(0.5f); }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr fp8_t infinity() { return fp8_t::bit_cast(0x80); }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr fp8_t quiet_NaN() { return fp8_t::bit_cast(0x80); }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr fp8_t signaling_NaN() { return fp8_t::bit_cast(0x80); }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr fp8_t denorm_min() { return fp8_t::bit_cast(0x01); }
};

template <>
struct numeric_limits<bf8_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr bf8_t min() { return bf8_t::bit_cast(0x04); }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr bf8_t lowest() { return bf8_t::bit_cast(0xff); }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr bf8_t max() { return bf8_t::bit_cast(0x7f); }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr bf8_t epsilon() { return bf8_t::bit_cast(0x34); }

    // maximum rounding error
    CK_TILE_HOST_DEVICE static constexpr bf8_t round_error() { return bf8_t(0.5f); }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr bf8_t infinity() { return bf8_t::bit_cast(0x80); }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr bf8_t quiet_NaN() { return bf8_t::bit_cast(0x80); }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr bf8_t signaling_NaN() { return bf8_t::bit_cast(0x80); }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr bf8_t denorm_min() { return bf8_t::bit_cast(0x01); }
};

CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, fp8_t)
CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, bf8_t)

// math
CK_TILE_HOST_DEVICE
fp8_t abs(const fp8_t& x) { return fp8_t::bit_cast(x.get() & 0x7f); }

CK_TILE_HOST_DEVICE
bool isnan(const fp8_t& x)
{
    uint8_t xx = x.get();
    return xx == 0x80; // TODO: NANOO
}

CK_TILE_DEVICE
fp8_t sqrt(fp8_t x) { return static_cast<fp8_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x))); };

CK_TILE_DEVICE
fp8_t exp(fp8_t x) { return static_cast<fp8_t>(__expf(static_cast<float>(x))); };

CK_TILE_DEVICE
fp8_t exp2(fp8_t x) { return static_cast<fp8_t>(exp2f(static_cast<float>(x))); };

CK_TILE_DEVICE
fp8_t log(fp8_t x) { return static_cast<fp8_t>(__logf(static_cast<float>(x))); };

CK_TILE_HOST_DEVICE
bf8_t abs(const bf8_t& x) { return bf8_t::bit_cast(x.get() & 0x7f); }

CK_TILE_HOST_DEVICE
bool isnan(const bf8_t& x)
{
    uint8_t xx = x.get();
    return xx == 0x80; // TODO: NANOO
}

CK_TILE_DEVICE
bf8_t sqrt(bf8_t x) { return static_cast<bf8_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x))); };

CK_TILE_DEVICE
bf8_t exp(bf8_t x) { return static_cast<bf8_t>(__expf(static_cast<float>(x))); };

CK_TILE_DEVICE
bf8_t exp2(bf8_t x) { return static_cast<bf8_t>(exp2f(static_cast<float>(x))); };

CK_TILE_DEVICE
bf8_t log(bf8_t x) { return static_cast<bf8_t>(__logf(static_cast<float>(x))); };

} // namespace ck_tile
