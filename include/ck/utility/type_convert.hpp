// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/f8_utils.hpp"
#include "ck/utility/random_gen.hpp"

namespace ck {
// Define the common macro for gfx94x models
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define __gfx94__
#endif

// Declare a template function for bf16 conversion using RTN
template <typename Y, typename X>
__host__ __device__ constexpr Y bf16_convert_rtn(X x);

// Convert fp32 to bf16 with RTN if higher precision is needed
template <>
inline __host__ __device__ constexpr bhalf_t bf16_convert_rtn<bhalf_t, float>(float x)
{
    // Nan check
    if(x != x)
    {
        return uint16_t(0x7FC0);
    }

    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    const uint32_t first_bf16_mantisa_bit = ((u.int32 >> 16) & 1);
    constexpr uint32_t rounding_bias      = uint32_t((1 << 15) - 1);

    return uint16_t((u.int32 + first_bf16_mantisa_bit + rounding_bias) >> 16);
}

// convert fp16 to bfp16 via fp32 with RTN if higher precision is needed
template <>
inline __host__ __device__ constexpr bhalf_t bf16_convert_rtn<bhalf_t, half_t>(half_t x)
{
    float x_fp32 = static_cast<float>(x);

    return bf16_convert_rtn<bhalf_t>(x_fp32);
}

// Convert X to Y, both X and Y are non-const data types.
template <typename Y,
          typename X,
          std::enable_if_t<!(std::is_const_v<Y> || std::is_const_v<X>), bool> = false>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    return static_cast<Y>(x);
}

// Convert X to Y, either X or Y is a const data type.
template <typename Y,
          typename X,
          std::enable_if_t<std::is_const_v<Y> || std::is_const_v<X>, bool> = false>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    using NonConstY = std::remove_const_t<Y>;
    using NonConstX = std::remove_const_t<X>;
    return static_cast<Y>(type_convert<NonConstY, NonConstX>(x));
}

// convert bfp16 to fp32
template <>
inline __host__ __device__ constexpr float type_convert<float, bhalf_t>(bhalf_t x)
{
    union
    {
        uint32_t int32;
        float fp32;
    } u = {uint32_t(x) << 16};

    return u.fp32;
}

// convert fp32 to bfp16, round to nearest even
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, float>(float x)
{
#if CK_USE_RNE_BF16_CONVERSION
    return bf16_convert_rtn<bhalf_t>(x);
#else
    return uint16_t(u.int32 >> 16);
#endif
}

// convert bfp16 to fp16 via fp32
template <>
inline __host__ __device__ constexpr half_t type_convert<half_t, bhalf_t>(bhalf_t x)
{
    float x_fp32 = type_convert<float>(x);

    return static_cast<half_t>(x_fp32);
}

// convert fp16 to bfp16 via fp32
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, half_t>(half_t x)
{
    float x_fp32 = static_cast<float>(x);

    return type_convert<bhalf_t>(x_fp32);
}

// convert bfp16 to int8 via fp32
template <>
inline __host__ __device__ constexpr int8_t type_convert<int8_t, bhalf_t>(bhalf_t x)
{
    float x_fp32 = type_convert<float>(x);

    return static_cast<int8_t>(x_fp32);
}

// convert int8 to bfp16 via fp32
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, int8_t>(int8_t x)
{
    float x_fp32 = static_cast<float>(x);

    return type_convert<bhalf_t>(x_fp32);
}

// Convert X to Y
template <typename Y, typename X>
__host__ __device__ constexpr Y type_convert_sp(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    return static_cast<Y>(x);
}

template <>
inline __host__ __device__ constexpr int type_convert_sp<int, float>(float x)
{
    union
    {
        float fp32;
        int int32;
    } u = {x};

    return u.int32;
}

template <>
inline __host__ __device__ constexpr float type_convert_sp<float, int>(int x)
{
    union
    {
        int int32;
        float fp32;
    } u = {x};

    return u.fp32;
}

template <>
inline __host__ __device__ constexpr int type_convert_sp<int, half_t>(half_t x)
{
    union
    {
        half_t fp16;
        int int32;
    } u = {x};

    return u.int32;
}

template <>
inline __host__ __device__ constexpr half_t type_convert_sp<half_t, int>(int x)
{
    union
    {
        int int32;
        half_t fp16;
    } u = {x};

    return u.fp16;
}

// Declare a template function for fp8 conversion using SR
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_sr(X x);

// convert fp32 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_t f8_convert_sr<f8_t, float>(float x)
{
    constexpr int seed = 1254739;
    uint32_t rng       = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_fp8 = 240.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_fp8, -max_fp8);
    ival       = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
    val.i32val = ival;
    return val.i8val[0]; // little endian
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    return utils::
        cast_to_f8<float, f8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(x,
                                                                                               rng);
#endif
}

// convert fp16 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_t f8_convert_sr<f8_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_sr<f8_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    constexpr int seed               = 1254739;
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<uintptr_t>(&x), x);
    return utils::
        cast_to_f8<half_t, f8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp32 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_t f8_convert_sr<bf8_t, float>(float x)
{
    constexpr int seed = 1254739;
    uint32_t rng       = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_bf8 = 57344.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_bf8, -max_bf8);
    ival       = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
    val.i32val = ival;
    return val.i8val[0]; // little endian
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    return utils::
        cast_to_f8<float, bf8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp16 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_t f8_convert_sr<bf8_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_sr<bf8_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    constexpr int seed               = 1254739;
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<uintptr_t>(&x), x);
    return utils::
        cast_to_f8<half_t, bf8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// Declare a template function for fp8 conversion using RNE
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_rne(X x);

// convert fp32 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_t f8_convert_rne<f8_t, float>(float x)
{
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_fp8 = 240.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_fp8, -max_fp8);
    ival       = __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival, false); // false -> WORD0
    val.i32val = ival;
    return val.i8val[0];
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::
        cast_to_f8<float, f8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(x,
                                                                                               rng);
#endif
}

// convert fp16 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_t f8_convert_rne<f8_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_rne<f8_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::
        cast_to_f8<half_t, f8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp32 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_t f8_convert_rne<bf8_t, float>(float x)
{
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_bf8 = 57344.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_bf8, -max_bf8);
    ival       = __builtin_amdgcn_cvt_pk_bf8_f32(val.fval, val.fval, ival, false); // false -> WORD0
    val.i32val = ival;
    return val.i8val[0];
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::
        cast_to_f8<float, bf8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp16 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_t f8_convert_rne<bf8_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_rne<bf8_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::
        cast_to_f8<half_t, bf8_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp32 to fp8
template <>
inline __host__ __device__ f8_t type_convert<f8_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_t>(x);
#else
    return f8_convert_rne<f8_t>(x);
#endif
}

// convert fp8 to fp32
template <>
inline __host__ __device__ float type_convert<float, f8_t>(f8_t x)
{
#if defined(__gfx94__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);
    fval            = __builtin_amdgcn_cvt_f32_fp8(i32val, 0);
    // asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<f8_t, float, negative_zero_nan>(x);
#endif
}

template <>
inline __host__ __device__ float2_t type_convert<float2_t, f8x2_t>(f8x2_t x)
{
#if defined(__gfx94__)
    const auto i16val = bit_cast<uint16_t>(x);
    return __builtin_amdgcn_cvt_pk_f32_fp8(i16val, 0);
#else
    constexpr bool negative_zero_nan = true;
    const auto f8x2_v                = vector_type<f8_t, 2>(x);
    vector_type<float, 2> f32x2_v;
    f32x2_v.template AsType<float>()(Number<0>{}) =
        utils::cast_from_f8<f8_t, float, negative_zero_nan>(
            f8x2_v.template AsType<f8_t>()[Number<0>{}]);
    f32x2_v.template AsType<float>()(Number<1>{}) =
        utils::cast_from_f8<f8_t, float, negative_zero_nan>(
            f8x2_v.template AsType<f8_t>()[Number<1>{}]);
    return f32x2_v.template AsType<float2_t>()[Number<0>{}];
#endif
}

template <>
inline __host__ __device__ half2_t type_convert<half2_t, float2_t>(float2_t x)
{

    const vector_type<float, 2> f32x2_v(x);
    const auto y = __builtin_amdgcn_cvt_pkrtz(f32x2_v.template AsType<float>()[Number<0>{}],
                                              f32x2_v.template AsType<float>()[Number<1>{}]);
    return bit_cast<half2_t>(y);
}

// convert fp16 to fp8
template <>
inline __host__ __device__ f8_t type_convert<f8_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_t>(x);
#else
    return f8_convert_rne<f8_t>(x);
#endif
}

// convert fp8 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, f8_t>(f8_t x)
{
#if defined(__gfx94__)
    // use native conversion to float and convert to fp16
    return type_convert<half_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<f8_t, half_t, negative_zero_nan>(x);
#endif
}

// convert fp32 to bf8
template <>
inline __host__ __device__ bf8_t type_convert<bf8_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_t>(x);
#else
    return f8_convert_rne<bf8_t>(x);
#endif
}

// convert bf8 to fp32
template <>
inline __host__ __device__ float type_convert<float, bf8_t>(bf8_t x)
{
#if defined(__gfx94__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);
    fval            = __builtin_amdgcn_cvt_f32_bf8(i32val, 0);
    // asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<bf8_t, float, negative_zero_nan>(x);
#endif
}

// convert fp16 to bf8
template <>
inline __host__ __device__ bf8_t type_convert<bf8_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_t>(x);
#else
    return f8_convert_rne<bf8_t>(x);
#endif
}

// convert bf8 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, bf8_t>(bf8_t x)
{
#if defined(__gfx94__)
    // use native conversion to float and convert to fp16
    return type_convert<half_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<bf8_t, half_t, negative_zero_nan>(x);
#endif
}

} // namespace ck
