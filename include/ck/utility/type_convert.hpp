// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/e8m0_utils.hpp"
#include "ck/utility/f8_utils.hpp"
#include "ck/utility/mxf4_utils.hpp"
#include "ck/utility/random_gen.hpp"
#include "ck/utility/array.hpp"

namespace ck {
// Define the common macro for MI300 models
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
#define __gfx94__
#endif

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

// convert fp32 to bfp16
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, float>(float x)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    return uint16_t(u.int32 >> 16);
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

// convert fp32 to fp4 with rounding to nearest even
inline __host__ __device__ f4_t f4_convert_rne(float x, float scale = 1.0f)
{
#if defined(__gfx950__)
    union
    {
        uint32_t bitwise;
        f4_t f4_array[4];
    } value{0};
    value.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(value.bitwise,
                                                 in.template AsType<float>()(Number<0>{}),
                                                 in.template AsType<float>()(Number<1>{}),
                                                 scale,
                                                 0);
    return value.f4_array[0];
#else
    return utils::sat_convert_to_type<f4_t>(x / scale);
#endif
}

// convert vector of 2 fp32 to vector of 2 fp4 with rne
inline __host__ __device__ f4x2_t f4_convert_rne(float2_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    union
    {
        uint32_t bitwise;
        fp4x2_t f4x2_array[4];
    } value{0};
    value.bitwise = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(value.bitwise, x[0], x[1], scale, 0);
    return value.f4x2_array[0];
#else
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};
    uint8_t l     = utils::sat_convert_to_type<f4_t>(x[1] / scale);
    uint8_t h     = utils::sat_convert_to_type<f4_t>(x[0] / scale);
    value.bitwise = (h << 4) | l;
    return value.f4x2_array[0];
#endif
}

// convert vector of 32 fp32 to vector of 32 fp4 with rne
inline __host__ __device__ f4x32_t f4_convert_rne(float32_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{}, tmp_values{};
    // TODO: pack in a loop
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[0], x[1], scale, 0);
    f4_values.f4x2_array[0] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[2], x[3], scale, 0);
    f4_values.f4x2_array[1] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[4], x[5], scale, 0);
    f4_values.f4x2_array[2] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[6], x[7], scale, 0);
    f4_values.f4x2_array[3] = tmp_values.f4x2_array[0];

    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[8], x[9], scale, 0);
    f4_values.f4x2_array[4] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[10], x[11], scale, 0);
    f4_values.f4x2_array[5] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[12], x[13], scale, 0);
    f4_values.f4x2_array[6] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[14], x[15], scale, 0);
    f4_values.f4x2_array[7] = tmp_values.f4x2_array[0];

    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[16], x[17], scale, 0);
    f4_values.f4x2_array[8] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[18], x[19], scale, 0);
    f4_values.f4x2_array[9] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[20], x[21], scale, 0);
    f4_values.f4x2_array[10] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[22], x[23], scale, 0);
    f4_values.f4x2_array[11] = tmp_values.f4x2_array[0];

    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[24], x[25], scale, 0);
    f4_values.f4x2_array[12] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[26], x[27], scale, 0);
    f4_values.f4x2_array[13] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[28], x[29], scale, 0);
    f4_values.f4x2_array[14] = tmp_values.f4x2_array[0];
    tmp_values.bitwise =
        __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(tmp_values.bitwise, x[30], x[31], scale, 0);
    f4_values.f4x2_array[15] = tmp_values.f4x2_array[0];

    return f4_values.f4x32_array;
#else
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{};
    // TODO: pack in a loop
    auto tmp = utils::sat_convert_to_type<f4_t>(x[0] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[1] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[2] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[3] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[4] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[5] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[6] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[7] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;

    tmp = utils::sat_convert_to_type<f4_t>(x[8] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[9] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[10] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[11] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[12] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[13] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[14] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[15] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;

    tmp = utils::sat_convert_to_type<f4_t>(x[16] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[17] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[18] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[19] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[20] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[21] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[22] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[23] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;

    tmp = utils::sat_convert_to_type<f4_t>(x[24] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[25] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[26] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[27] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[28] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[29] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[30] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type<f4_t>(x[31] / scale);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;

    return f4_values.f4x32_array;
#endif
}

// convert fp32 to fp4 with stochastic rounding
inline __host__ __device__ f4_t f4_convert_sr(float x, float scale = 1.0f)
{
    constexpr int seed = 1254739;
    uint32_t rng       = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#if defined(__gfx950__)
    union
    {
        uint32_t bitwise;
        f4_t f4_array[4];
    } value{0};
    value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(value.bitwise, x, rng, scale, 0);
    return value.f4_array[0];
#else
    return utils::sat_convert_to_type_sr<f4_t>(x / scale, rng);
#endif
}

// convert vector of 2 fp32 to vector of 2 fp4 with sr
inline __host__ __device__ f4x2_t f4_convert_sr(float2_t x, float scale = 1.0f)
{
    constexpr int seed = 1254739;
    uint32_t rng       = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x[0]);
#if defined(__gfx950__)
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};
    value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(value.bitwise, x, rng, scale, 0);
    return value.f4x2_array[0];
#else
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};
    uint8_t l     = utils::sat_convert_to_type_sr<f4_t>(x[1] / scale, rng);
    uint8_t h     = utils::sat_convert_to_type_sr<f4_t>(x[0] / scale, rng);
    value.bitwise = (h << 4) | l;
    return value.f4x2_array[0];
#endif
}

// convert vector of 32 fp32 to vector of 32 fp4 with sr
inline __host__ __device__ f4x32_t f4_convert_sr(float32_t x, float scale = 1.0f)
{
    constexpr int seed = 1254739;
    uint32_t rng       = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x[0]);
#if defined(__gfx950__)
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{0}, tmp_values{0};
    union
    {
        float2_t floatx2_array[16];
        float32_t floatx32_array;
    } float_values{0};
    // TODO: pack in a loop
    tmp_values.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[0], rng, scale, 0);
    f4_values.f4x2_array[0] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[1], rng, scale, 0);
    f4_values.f4x2_array[1] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[2], rng, scale, 0);
    f4_values.f4x2_array[2] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[3], rng, scale, 0);
    f4_values.f4x2_array[3] = tmp_values.f4x2_array[0];

    tmp_values.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[4], rng, scale, 0);
    f4_values.f4x2_array[4] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[5], rng, scale, 0);
    f4_values.f4x2_array[5] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[6], rng, scale, 0);
    f4_values.f4x2_array[6] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[7], rng, scale, 0);
    f4_values.f4x2_array[7] = tmp_values.f4x2_array[0];

    tmp_values.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[8], rng, scale, 0);
    f4_values.f4x2_array[8] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[9], rng, scale, 0);
    f4_values.f4x2_array[9] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[10], rng, scale, 0);
    f4_values.f4x2_array[10] = tmp_values.f4x2_array[0];
    tmp_values.bitwise       = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[11], rng, scale, 0);
    f4_values.f4x2_array[11] = tmp_values.f4x2_array[0];

    tmp_values.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[12], rng, scale, 0);
    f4_values.f4x2_array[12] = tmp_values.f4x2_array[0];
    tmp_values.bitwise       = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[13], rng, scale, 0);
    f4_values.f4x2_array[13] = tmp_values.f4x2_array[0];
    tmp_values.bitwise       = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[14], rng, scale, 0);
    f4_values.f4x2_array[14] = tmp_values.f4x2_array[0];
    tmp_values.bitwise       = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.floatx2_array[15], rng, scale, 0);
    f4_values.f4x2_array[15] = tmp_values.f4x2_array[0];

    return f4_values.f4x32_array;
#else
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{0};
    // TODO: pack in a loop
    auto tmp = utils::sat_convert_to_type_sr<f4_t>(x[0] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[1] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[2] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[3] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[4] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[5] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[6] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[7] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;

    tmp = utils::sat_convert_to_type_sr<f4_t>(x[8] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[9] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[10] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[11] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[12] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[13] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[14] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[15] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;

    tmp = utils::sat_convert_to_type_sr<f4_t>(x[16] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[17] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[18] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[19] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[20] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[21] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[22] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[23] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;

    tmp = utils::sat_convert_to_type_sr<f4_t>(x[24] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[25] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[26] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[27] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[28] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[29] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[30] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;
    tmp = utils::sat_convert_to_type_sr<f4_t>(x[31] / scale, rng);
    f4_values.bitwise <<= 4;
    f4_values.bitwise |= tmp;

    return f4_values.f4x32_array;
#endif
}

// convert fp32 to fp4
template <>
inline __host__ __device__ f4_t type_convert<f4_t, float>(float x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

// convert vector of 2 fp32 to vector of 2 fp4
template <>
inline __host__ __device__ f4x2_t type_convert<f4x2_t, float2_t>(float2_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

// convert vector of 32 fp32 to vector of 32 fp4
template <>
inline __host__ __device__ f4x32_t type_convert<f4x32_t, float32_t>(float32_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

// convert fp4 to fp32
template <>
inline __host__ __device__ float type_convert<float, f4_t>(f4_t x)
{
#if defined(__gfx950__)
    float scale = 1.0f;
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, scale, 0)
        .template AsType<float>()(Number<0>{});
#else
    return utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(), x);
#endif
}

// convert vector of 2 fp4 to vector of 2 fp32
template <>
inline __host__ __device__ float2_t type_convert<float2_t, f4x2_t>(f4x2_t x)
{
#if defined(__gfx950__)
    float scale = 1.0f;
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, scale, 0);
#else
    float2_t ret{utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(), x.unpack(1)),
                 utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(), x.unpack(0))};
    return ret;
#endif
}

// convert vector of 32 fp4 to vector of 32 fp32
template <>
inline __host__ __device__ float32_t type_convert<float32_t, f4x32_t>(f4x32_t x)
{
#if defined(__gfx950__)
    union
    {
        f4x32_t f4x32_array;
        f4x2_t fp4x2[16];
    } value{x};
    float2_t op;
    float32_t ret;
    float scale = 1.0f;
    // TODO: pack in a loop
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[0], type_convert<float>(scale), 0);
    ret[0] = op[0];
    ret[1] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[1], type_convert<float>(scale), 0);
    ret[2] = op[0];
    ret[3] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[2], type_convert<float>(scale), 0);
    ret[4] = op[0];
    ret[5] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[3], type_convert<float>(scale), 0);
    ret[6] = op[0];
    ret[7] = op[1];

    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[4], type_convert<float>(scale), 0);
    ret[8] = op[0];
    ret[9] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[5], type_convert<float>(scale), 0);
    ret[10] = op[0];
    ret[11] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[6], type_convert<float>(scale), 0);
    ret[12] = op[0];
    ret[13] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[7], type_convert<float>(scale), 0);
    ret[14] = op[0];
    ret[15] = op[1];

    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[8], type_convert<float>(scale), 0);
    ret[16] = op[0];
    ret[17] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[9], type_convert<float>(scale), 0);
    ret[18] = op[0];
    ret[19] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[10], type_convert<float>(scale), 0);
    ret[20] = op[0];
    ret[21] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[11], type_convert<float>(scale), 0);
    ret[22] = op[0];
    ret[23] = op[1];

    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[12], type_convert<float>(scale), 0);
    ret[24] = op[0];
    ret[25] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[13], type_convert<float>(scale), 0);
    ret[26] = op[0];
    ret[27] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[14], type_convert<float>(scale), 0);
    ret[28] = op[0];
    ret[29] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[15], type_convert<float>(scale), 0);
    ret[30] = op[0];
    ret[31] = op[1];

    return ret;
#else
    union
    {
        float32_t float32_array;
        float float_array[32];
    } float_values{};
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{bit_cast<__uint128_t>(x)};
    // TODO: pack in a loop
    float_values.float_array[0] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[0].unpack(0));
    float_values.float_array[1] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[0].unpack(1));
    float_values.float_array[2] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[1].unpack(0));
    float_values.float_array[3] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[1].unpack(1));
    float_values.float_array[4] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[2].unpack(0));
    float_values.float_array[5] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[2].unpack(1));
    float_values.float_array[6] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[3].unpack(0));
    float_values.float_array[7] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[3].unpack(1));

    float_values.float_array[0] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[4].unpack(0));
    float_values.float_array[1] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[4].unpack(1));
    float_values.float_array[2] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[5].unpack(0));
    float_values.float_array[3] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[5].unpack(1));
    float_values.float_array[4] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[6].unpack(0));
    float_values.float_array[5] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[6].unpack(1));
    float_values.float_array[6] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[7].unpack(0));
    float_values.float_array[7] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[7].unpack(1));

    float_values.float_array[0] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[8].unpack(0));
    float_values.float_array[1] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[8].unpack(1));
    float_values.float_array[2] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[9].unpack(0));
    float_values.float_array[3] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[9].unpack(1));
    float_values.float_array[4] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[10].unpack(0));
    float_values.float_array[5] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[10].unpack(1));
    float_values.float_array[6] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[11].unpack(0));
    float_values.float_array[7] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[11].unpack(1));

    float_values.float_array[0] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[12].unpack(0));
    float_values.float_array[1] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[12].unpack(1));
    float_values.float_array[2] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[13].unpack(0));
    float_values.float_array[3] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[13].unpack(1));
    float_values.float_array[4] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[14].unpack(0));
    float_values.float_array[5] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[14].unpack(1));
    float_values.float_array[6] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[15].unpack(0));
    float_values.float_array[7] = utils::to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(),
                                                        f4_values.f4x2_array[15].unpack(1));

    return float_values.float32_array;
#endif
}

template <>
inline __host__ __device__ float type_convert<float, e8m0_scale_t>(e8m0_scale_t scale)
{
    return utils::cast_to_float(scale);
}

template <>
inline __host__ __device__ e8m0_scale_t type_convert<e8m0_scale_t, float>(float scale)
{
    return utils::cast_from_float(scale);
}

// Declare a template function for scaled conversion
template <typename Y, typename X>
__host__ __device__ constexpr Y scaled_type_convert(e8m0_scale_t scale, X x);

// convert fp4 to fp32
template <>
inline __host__ __device__ float scaled_type_convert<float, f4_t>(e8m0_scale_t scale, f4_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, type_convert<float>(scale), 0)
        .template AsType<float>()(Number<0>{});
#else
    return utils::to_float<f4_t>(scale, x);
#endif
}

// convert vector of 2 fp4 to vector of 2 fp32
template <>
inline __host__ __device__ float2_t scaled_type_convert<float2_t, f4x2_t>(e8m0_scale_t scale,
                                                                          f4x2_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, type_convert<float>(scale), 0);
#else
    float2_t ret{utils::to_float<f4_t>(scale, x.unpack(1)),
                 utils::to_float<f4_t>(scale, x.unpack(0))};
    return ret;
#endif
}

// convert vector of 32 fp4 to vector of 32 fp32
template <>
inline __host__ __device__ float32_t scaled_type_convert<float32_t, f4x32_t>(e8m0_scale_t scale,
                                                                             f4x32_t x)
{
#if defined(__gfx950__)
    union
    {
        f4x32_t f4x32_array;
        f4x2_t fp4x2[16];
    } value{x};
    float2_t op;
    float32_t ret;
    // TODO: pack in a loop
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[0], type_convert<float>(scale), 0);
    ret[0] = op[0];
    ret[1] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[1], type_convert<float>(scale), 0);
    ret[2] = op[0];
    ret[3] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[2], type_convert<float>(scale), 0);
    ret[4] = op[0];
    ret[5] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[3], type_convert<float>(scale), 0);
    ret[6] = op[0];
    ret[7] = op[1];

    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[4], type_convert<float>(scale), 0);
    ret[8] = op[0];
    ret[9] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[5], type_convert<float>(scale), 0);
    ret[10] = op[0];
    ret[11] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[6], type_convert<float>(scale), 0);
    ret[12] = op[0];
    ret[13] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[7], type_convert<float>(scale), 0);
    ret[14] = op[0];
    ret[15] = op[1];

    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[8], type_convert<float>(scale), 0);
    ret[16] = op[0];
    ret[17] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[9], type_convert<float>(scale), 0);
    ret[18] = op[0];
    ret[19] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[10], type_convert<float>(scale), 0);
    ret[20] = op[0];
    ret[21] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[11], type_convert<float>(scale), 0);
    ret[22] = op[0];
    ret[23] = op[1];

    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[12], type_convert<float>(scale), 0);
    ret[24] = op[0];
    ret[25] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[13], type_convert<float>(scale), 0);
    ret[26] = op[0];
    ret[27] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[14], type_convert<float>(scale), 0);
    ret[28] = op[0];
    ret[29] = op[1];
    op = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[15], type_convert<float>(scale), 0);
    ret[30] = op[0];
    ret[31] = op[1];

    return ret;
#else
    union
    {
        float32_t float32_array;
        float float_array[32];
    } float_values{};
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{bit_cast<__uint128_t>(x)};
    // TODO: pack in a loop
    float_values.float_array[0] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[0].unpack(0));
    float_values.float_array[1] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[0].unpack(1));
    float_values.float_array[2] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[1].unpack(0));
    float_values.float_array[3] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[1].unpack(1));
    float_values.float_array[4] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[2].unpack(0));
    float_values.float_array[5] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[2].unpack(1));
    float_values.float_array[6] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[3].unpack(0));
    float_values.float_array[7] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[3].unpack(1));

    float_values.float_array[0] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[4].unpack(0));
    float_values.float_array[1] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[4].unpack(1));
    float_values.float_array[2] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[5].unpack(0));
    float_values.float_array[3] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[5].unpack(1));
    float_values.float_array[4] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[6].unpack(0));
    float_values.float_array[5] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[6].unpack(1));
    float_values.float_array[6] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[7].unpack(0));
    float_values.float_array[7] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[7].unpack(1));

    float_values.float_array[0] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[8].unpack(0));
    float_values.float_array[1] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[8].unpack(1));
    float_values.float_array[2] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[9].unpack(0));
    float_values.float_array[3] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[9].unpack(1));
    float_values.float_array[4] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[10].unpack(0));
    float_values.float_array[5] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[10].unpack(1));
    float_values.float_array[6] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[11].unpack(0));
    float_values.float_array[7] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[11].unpack(1));

    float_values.float_array[0] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[12].unpack(0));
    float_values.float_array[1] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[12].unpack(1));
    float_values.float_array[2] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[13].unpack(0));
    float_values.float_array[3] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[13].unpack(1));
    float_values.float_array[4] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[14].unpack(0));
    float_values.float_array[5] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[14].unpack(1));
    float_values.float_array[6] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[15].unpack(0));
    float_values.float_array[7] = utils::to_float<f4_t>(scale, f4_values.f4x2_array[15].unpack(1));

    return float_values.float32_array;
#endif
}

// convert fp32 to fp4
template <>
inline __host__ __device__ f4_t scaled_type_convert<f4_t, float>(e8m0_scale_t scale, float x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

// convert vector of 2 fp32 to vector of 2 fp4
template <>
inline __host__ __device__ f4x2_t scaled_type_convert<f4x2_t, float2_t>(e8m0_scale_t scale,
                                                                        float2_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

// convert vector of 32 fp32 to vector of 32 fp4
template <>
inline __host__ __device__ f4x32_t scaled_type_convert<f4x32_t, float32_t>(e8m0_scale_t scale,
                                                                           float32_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

template <typename Y, typename X, std::size_t NumElems>
inline __host__ __device__ void array_convert(std::array<Y, NumElems>& y,
                                              const std::array<X, NumElems>& x)
{
    for(std::size_t i = 0; i < NumElems; i++)
    {
        y[i] = type_convert<Y>(x[i]);
    }
}

template <typename Y, typename X, index_t NumElems>
inline __host__ __device__ void array_convert(Array<Y, NumElems>& y, const Array<X, NumElems>& x)
{
    for(std::size_t i = 0; i < NumElems; i++)
    {
        y[i] = type_convert<Y>(x[i]);
    }
}

// Declare a template function for bf16 conversion using RTN
template <typename Y, typename X>
__host__ __device__ constexpr Y bf16_convert_rtn(X x);

// Convert fp32 to bf16 with RTN if higher precision is needed
template <>
inline __host__ __device__ constexpr bhalf_t bf16_convert_rtn<bhalf_t, float>(float x)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

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
    // with an exponent of 0x00 and a mantissa of 0x7f, it may be rounded up
    // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
    // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
    // incrementing it causes it to become an exponent of 0xFF and a mantissa
    // of 0x00, which is Inf, the next higher value to the unrounded value.
    bool flag0 = ~u.int32 & 0x7f800000;

    // When all of the exponent bits are 1, the value is Inf or NaN.
    // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
    // mantissa bit. Quiet NaN is indicated by the most significant mantissa
    // bit being 1. Signaling NaN is indicated by the most significant
    // mantissa bit being 0 but some other bit(s) being 1. If any of the
    // lower 16 bits of the mantissa are 1, we set the least significant bit
    // of the bfloat16 mantissa, in order to preserve signaling NaN in case
    // the bfloat16's mantissa bits are all 0.
    bool flag1 = !flag0 && (u.int32 & 0xffff);

    u.int32 += flag0 ? 0x7fff + ((u.int32 >> 16) & 1) : 0; // Round to nearest, round to even
    u.int32 |= flag1 ? 0x10000 : 0x0;                      // Preserve signaling NaN

    return uint16_t(u.int32 >> 16);
}

// convert fp16 to bfp16 via fp32 with RTN if higher precision is needed
template <>
inline __host__ __device__ constexpr bhalf_t bf16_convert_rtn<bhalf_t, half_t>(half_t x)
{
    float x_fp32 = static_cast<float>(x);

    return bf16_convert_rtn<bhalf_t>(x_fp32);
}
} // namespace ck
