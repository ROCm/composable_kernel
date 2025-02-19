// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/f8_utils.hpp"
#include "ck/utility/mxf4_utils.hpp"
#include "ck/utility/mxf6_utils.hpp"
#include "ck/utility/random_gen.hpp"
#include "ck/utility/array.hpp"
#include "ck/utility/amd_inline_asm.hpp"
#include "ck/utility/type.hpp"

namespace ck {
// Define the common macro for MI300 models
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
#define __gfx94__
#endif

namespace {
namespace details {

[[maybe_unused]] __host__ half2_t pk_add_f16(const half2_t& x, const half2_t& y)
{
    half2_t vector_res;

    vector_res.x = x.x + y.x;
    vector_res.y = x.y + y.y;

    return vector_res;
}

[[maybe_unused]] __device__ half2_t pk_add_f16(const half2_t& x, const half2_t& y)
{
    return amd_assembly_pk_add_f16(x, y);
}
} // namespace details
} // namespace

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
          ck::enable_if_t<!(ck::is_const_v<Y> || ck::is_const_v<X>), bool> = false>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!ck::is_reference_v<Y> && !ck::is_reference_v<X>);

    return static_cast<Y>(x);
}

// Convert X to Y, either X or Y is a const data type.
template <typename Y,
          typename X,
          ck::enable_if_t<ck::is_const_v<Y> || ck::is_const_v<X>, bool> = false>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!ck::is_reference_v<Y> && !ck::is_reference_v<X>);

    using NonConstY = ck::remove_const_t<Y>;
    using NonConstX = ck::remove_const_t<X>;
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

template <>
inline __host__ __device__ constexpr f8_ocp_t type_convert<f8_ocp_t, int>(int x)
{
    return f8_ocp_t{type_convert<f8_ocp_t::data_type>(x)};
}

template <>
inline __host__ __device__ constexpr bf8_ocp_t type_convert<bf8_ocp_t, int>(int x)
{
    return bf8_ocp_t{type_convert<bf8_ocp_t::data_type>(x)};
}

// Convert X to Y
template <typename Y, typename X>
__host__ __device__ constexpr Y type_convert_sp(X x)
{
    static_assert(!ck::is_reference_v<Y> && !ck::is_reference_v<X>);

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
inline __host__ __device__ f8_fnuz_t f8_convert_sr<f8_fnuz_t, float>(float x)
{
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x);
#endif
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
        cast_to_f8<float, f8_fnuz_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp16 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_fnuz_t f8_convert_sr<f8_fnuz_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_sr<f8_fnuz_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    constexpr int seed               = 1254739;

#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<size_t>(&x), x);
#endif
    return utils::cast_to_f8<half_t,
                             f8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

// convert fp32 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_fnuz_t f8_convert_sr<bf8_fnuz_t, float>(float x)
{
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x);
#endif
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
    return utils::cast_to_f8<float,
                             bf8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

// convert fp16 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_fnuz_t f8_convert_sr<bf8_fnuz_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_sr<bf8_fnuz_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    constexpr int seed               = 1254739;

#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<size_t>(&x), x);
#endif
    return utils::cast_to_f8<half_t,
                             bf8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

// Declare a template function for fp8 conversion using RNE
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_rne(X x);

// convert fp32 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_fnuz_t f8_convert_rne<f8_fnuz_t, float>(float x)
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
        cast_to_f8<float, f8_fnuz_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp16 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_fnuz_t f8_convert_rne<f8_fnuz_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_rne<f8_fnuz_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::cast_to_f8<half_t,
                             f8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

// convert fp32 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_fnuz_t f8_convert_rne<bf8_fnuz_t, float>(float x)
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
    return utils::cast_to_f8<float,
                             bf8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

// convert fp16 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_fnuz_t f8_convert_rne<bf8_fnuz_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_rne<bf8_fnuz_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::cast_to_f8<half_t,
                             bf8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

// convert fp32 to fp8
template <>
inline __host__ __device__ f8_fnuz_t type_convert<f8_fnuz_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_fnuz_t>(x);
#else
    return f8_convert_rne<f8_fnuz_t>(x);
#endif
}

// convert fp32 to fp8
template <>
inline __host__ __device__ f8_ocp_t type_convert<f8_ocp_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_ocp_t>(x);
#else
    return f8_convert_rne<f8_ocp_t>(x);
#endif
}

// convert fp8 to fp32
template <>
inline __host__ __device__ float type_convert<float, f8_fnuz_t>(f8_fnuz_t x)
{
#if defined(__gfx94__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);
    fval            = __builtin_amdgcn_cvt_f32_fp8(i32val, 0);
    // asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<f8_fnuz_t, float, negative_zero_nan>(x);
#endif
}

template <>
inline __host__ __device__ float2_t type_convert<float2_t, f8x2_fnuz_t>(f8x2_fnuz_t x)
{
#if defined(__gfx94__)
    const auto i16val = bit_cast<uint16_t>(x);
    return __builtin_amdgcn_cvt_pk_f32_fp8(i16val, 0);
#else
    constexpr bool negative_zero_nan = true;
    const auto f8x2_v                = vector_type<f8_fnuz_t, 2>(x);
    vector_type<float, 2> f32x2_v;
    f32x2_v.template AsType<float>()(Number<0>{}) =
        utils::cast_from_f8<f8_fnuz_t, float, negative_zero_nan>(
            f8x2_v.template AsType<f8_fnuz_t>()[Number<0>{}]);
    f32x2_v.template AsType<float>()(Number<1>{}) =
        utils::cast_from_f8<f8_fnuz_t, float, negative_zero_nan>(
            f8x2_v.template AsType<f8_fnuz_t>()[Number<1>{}]);
    return f32x2_v.template AsType<float2_t>()[Number<0>{}];
#endif
}

template <>
inline __host__ __device__ float2_t type_convert<float2_t, f8x2_ocp_t>(f8x2_ocp_t x)
{
#if CK_OCP_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f32x2_from_f8x2<f8_ocp_t::default_interpret>(
        x.AsType<fp8_impl::fp8x2_storage_t>()[Number<0>{}]);
#else
    return float2_t{fp8_impl::cast_from_f8<float, f8_ocp_t::wm, f8_ocp_t::we, false>(
                        x.AsType<fp8_storage_t>()[Number<0>{}]),
                    fp8_impl::cast_from_f8<float, f8_ocp_t::wm, f8_ocp_t::we, false>(
                        x.AsType<fp8_storage_t>()[Number<1>{}])};
#endif
}

template <>
inline __host__ __device__ float2_t type_convert<float2_t, pk_i4_t>(pk_i4_t x)
{
    uint8_t x_u8 = ck::bit_cast<uint8_t>(x);

    float x_l = ((x_u8 & 0x0f) >> 0) - 8.f;
    float x_h = ((x_u8 & 0xf0) >> 4) - 8.f;

#ifdef CK_USE_PK4_LAYOUT_SHUFFLE
    float2_t res = {x_h, x_l};
#elif
    float2_t res = {x_l, x_h};
#endif
    return res;
}

template <>
inline __host__ __device__ half2_t type_convert<half2_t, pk_i4_t>(pk_i4_t x)
{
    uint8_t x_u8 = ck::bit_cast<uint8_t>(x);
#ifdef CK_USE_PK4_LAYOUT_SHUFFLE
    uint32_t i4s = ((x_u8 & 0x0f) << 16) | ((x_u8 & 0xf0) >> 4);
#else
    uint32_t i4s = ((x_u8 & 0xf0) << 12) | (x_u8 & 0xf);
#endif

    const int EX  = 0x64006400;
    const int SUB = 0xE408E408; //-8

    int lo = i4s | EX;

    return details::pk_add_f16(bit_cast<half2_t>(lo), bit_cast<half2_t>(SUB));
}

template <>
inline __host__ __device__ bhalf2_t type_convert<bhalf2_t, pk_i4_t>(pk_i4_t x)
{
    uint8_t x_u8 = ck::bit_cast<uint8_t>(x);

    float x_l = ((x_u8 & 0x0f) >> 0) - 8.f;
    float x_h = ((x_u8 & 0xf0) >> 4) - 8.f;

#ifdef CK_USE_PK4_LAYOUT_SHUFFLE
    bhalf2_t res = {type_convert<bhalf_t>(x_h), type_convert<bhalf_t>(x_l)};
#else
    bhalf2_t res = {type_convert<bhalf_t>(x_l), type_convert<bhalf_t>(x_h)};
#endif

    return res;
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
inline __host__ __device__ f8_fnuz_t type_convert<f8_fnuz_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_fnuz_t>(x);
#else
    return f8_convert_rne<f8_fnuz_t>(x);
#endif
}

// convert fp16 to fp8
template <>
inline __host__ __device__ f8_ocp_t type_convert<f8_ocp_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_ocp_t>(x);
#else
    return f8_convert_rne<f8_ocp_t>(x);
#endif
}

// convert fp8 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, f8_fnuz_t>(f8_fnuz_t x)
{
#if defined(__gfx94__)
    // use native conversion to float and convert to fp16
    return type_convert<half_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<f8_fnuz_t, half_t, negative_zero_nan>(x);
#endif
}

// convert fp32 to bf8
template <>
inline __host__ __device__ bf8_fnuz_t type_convert<bf8_fnuz_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_fnuz_t>(x);
#else
    return f8_convert_rne<bf8_fnuz_t>(x);
#endif
}

// convert fp32 to bf8
template <>
inline __host__ __device__ bf8_ocp_t type_convert<bf8_ocp_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_ocp_t>(x);
#else
    return f8_convert_rne<bf8_ocp_t>(x);
#endif
}

// convert bf8 to fp32
template <>
inline __host__ __device__ float type_convert<float, bf8_fnuz_t>(bf8_fnuz_t x)
{
#if defined(__gfx94__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);
    fval            = __builtin_amdgcn_cvt_f32_bf8(i32val, 0);
    // asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<bf8_fnuz_t, float, negative_zero_nan>(x);
#endif
}

// convert fp16 to bf8
template <>
inline __host__ __device__ bf8_fnuz_t type_convert<bf8_fnuz_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_fnuz_t>(x);
#else
    return f8_convert_rne<bf8_fnuz_t>(x);
#endif
}

// convert fp16 to bf8
template <>
inline __host__ __device__ bf8_ocp_t type_convert<bf8_ocp_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_ocp_t>(x);
#else
    return f8_convert_rne<bf8_ocp_t>(x);
#endif
}

// convert bf8 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, bf8_fnuz_t>(bf8_fnuz_t x)
{
#if defined(__gfx94__)
    // use native conversion to float and convert to fp16
    return type_convert<half_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<bf8_fnuz_t, half_t, negative_zero_nan>(x);
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
    value.bitwise = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(value.bitwise, x, x, scale, 0);
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
        f4x2_t f4x2_array[4];
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
    union
    {
        float float_array[2];
        float2_t float2_array;
    } float_values{{x}};

    value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.float2_array, rng, scale, 0);
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
    } float_values{{0}};
    // TODO: pack in a loop
    tmp_values.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[0], rng, scale, 0);
    f4_values.f4x2_array[0] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[1], rng, scale, 0);
    f4_values.f4x2_array[1] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[2], rng, scale, 0);
    f4_values.f4x2_array[2] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[3], rng, scale, 0);
    f4_values.f4x2_array[3] = tmp_values.f4x2_array[0];

    tmp_values.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[4], rng, scale, 0);
    f4_values.f4x2_array[4] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[5], rng, scale, 0);
    f4_values.f4x2_array[5] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[6], rng, scale, 0);
    f4_values.f4x2_array[6] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[7], rng, scale, 0);
    f4_values.f4x2_array[7] = tmp_values.f4x2_array[0];

    tmp_values.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[8], rng, scale, 0);
    f4_values.f4x2_array[8] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[9], rng, scale, 0);
    f4_values.f4x2_array[9] = tmp_values.f4x2_array[0];
    tmp_values.bitwise      = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[10], rng, scale, 0);
    f4_values.f4x2_array[10] = tmp_values.f4x2_array[0];
    tmp_values.bitwise       = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[11], rng, scale, 0);
    f4_values.f4x2_array[11] = tmp_values.f4x2_array[0];

    tmp_values.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[12], rng, scale, 0);
    f4_values.f4x2_array[12] = tmp_values.f4x2_array[0];
    tmp_values.bitwise       = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[13], rng, scale, 0);
    f4_values.f4x2_array[13] = tmp_values.f4x2_array[0];
    tmp_values.bitwise       = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[14], rng, scale, 0);
    f4_values.f4x2_array[14] = tmp_values.f4x2_array[0];
    tmp_values.bitwise       = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        tmp_values.bitwise, float_values.floatx2_array[15], rng, scale, 0);
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
    union
    {
        float float_array[2];
        float2_t float2_array;
    } float_values{};
    float scale               = 1.0f;
    float_values.float2_array = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, scale, 0);
    return float_values.float_array[0];
#else
    return utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x);
#endif
}

// convert vector of 2 fp4 to vector of 2 fp32
template <>
inline __host__ __device__ float2_t type_convert<float2_t, f4x2_t>(f4x2_t x)
{
#if defined(__gfx950__)
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{};
    value.f4x2_array[0] = x;
    float scale         = 1.0f;
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.bitwise, scale, 0);
#else
    float2_t ret{
        utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(),
                              x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{})),
        utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(),
                              x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}))};
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
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } bitwise_value{};
    float2_t op;
    float32_t ret;
    float scale = 1.0f;
    // TODO: pack in a loop
    bitwise_value.f4x2_array[0] = value.fp4x2[0];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[0] = op[0];
    ret[1] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[1];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[2] = op[0];
    ret[3] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[2];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[4] = op[0];
    ret[5] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[3];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[6] = op[0];
    ret[7] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[4];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[8] = op[0];
    ret[9] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[5];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[10] = op[0];
    ret[11] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[6];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[12] = op[0];
    ret[13] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[7];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[14] = op[0];
    ret[15] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[8];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[16] = op[0];
    ret[17] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[9];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[18] = op[0];
    ret[19] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[10];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[20] = op[0];
    ret[21] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[11];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[22] = op[0];
    ret[23] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[12];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[24] = op[0];
    ret[25] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[13];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[26] = op[0];
    ret[27] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[14];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
    ret[28] = op[0];
    ret[29] = op[1];

    bitwise_value.f4x2_array[0] = value.fp4x2[15];
    op                          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
        bitwise_value.bitwise, type_convert<float>(scale), 0);
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
    float_values.float_array[0] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[0].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[1] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[0].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[2] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[1].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[3] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[1].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[4] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[2].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[5] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[2].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[6] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[3].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[7] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[3].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));

    float_values.float_array[0] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[4].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[1] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[4].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[2] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[5].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[3] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[5].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[4] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[6].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[5] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[6].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[6] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[7].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[7] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[7].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));

    float_values.float_array[0] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[8].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[1] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[8].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[2] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[9].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[3] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[9].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[4] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[10].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[5] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[10].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[6] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[11].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[7] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[11].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));

    float_values.float_array[0] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[12].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[1] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[12].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[2] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[13].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[3] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[13].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[4] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[14].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[5] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[14].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[6] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[15].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[7] = utils::to_float<f4_t>(
        NumericLimits<e8m0_bexp_t>::Binary_1(),
        f4_values.f4x2_array[15].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));

    return float_values.float32_array;
#endif
}

/**
 * @brief Converts a float to a 6-bit float type (f6_t) using round-to-nearest-even.
 *
 * Divides the input by the specified scale, then saturates and converts it
 * to the 6-bit floating-point format (f6_t).
 *
 * @param x     The input float value.
 * @param scale A scaling factor applied to `x` before conversion.
 * @return      The converted f6_t value.
 */
inline __host__ __device__ f6_t f6_convert_rne(float x, float scale = 1.0f)
{
#if defined(__gfx950__)
    float16_t in1{x};
    float16_t in2{};

    union
    {
        f6x32_t f6_vector;
        f6_t f6_array[32];
    } out{};

    out.f6_vector = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, scale);

    return out.f6_array[0];
#else
    return utils::sat_convert_to_type<f6_t>(x / scale);
#endif
}

/**
 * @brief Converts a 32-element single-precision float array into a packed 6-bit representation.
 *
 * This function divides each input float by the provided scale value, then performs conversion with
 * rounding to nearest / even to pack each element into 6 bits of precision.
 *
 * @param x     A vector of 32 floats stored in float32_t.
 * @param scale A scaling factor for each float before conversion.
 * @return An f6x32_t object storing the compressed 6-bit representation.
 */
inline __host__ __device__ f6x32_t f6_convert_rne(float32_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    float16_t* in1 = reinterpret_cast<float16_t*>(&x);
    float16_t* in2 = reinterpret_cast<float16_t*>(&x + 16);
    return __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(*in1, *in2, scale);
#else
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    union
    {
        f6x32_t f6_vector;
        f6_t f6_array[32];
    } out{};

    ck::static_for<0, 32, 1>{}([&](auto i) {
        out.f6_array[i] = utils::sat_convert_to_type<f6_t>(in.float_array[i] / scale);
    });

    return out.f6_vector;
#endif
}

/**
 * @brief Converts a float to the 6-bit floating-point type (f6_t) using stochastic rounding.
 *
 * Divides the input by the specified scale, then performs saturation and conversion
 * to f6_t based on a pseudo-randomly generated seed.
 *
 * @param x     The input float value.
 * @param scale A scaling factor applied to `x` before conversion.
 * @return      The converted f6_t value.
 */
inline __host__ __device__ f6_t f6_convert_sr(float x, float scale = 1.0f)
{
    constexpr int seed = 1254739;
    uint32_t rng       = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#if defined(__gfx950__)
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    union
    {
        f6x32_t f6_vector;
        f6_t f6_array[32];
    } out{};

    out.f6_vector = __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32(in.float_vector, rng, scale);

    return out.f6_array[0];
#else
    return utils::sat_convert_to_type_sr<f6_t>(x / scale, rng);
#endif
}

/**
 * @brief Converts a 32-element single-precision float array into a packed 6-bit representation.
 *
 * This function divides each input float by the provided scale value, then performs conversion with
 * stochastic rounding to pack each element into 6 bits of precision.
 *
 * @param x     A vector of 32 floats stored in float32_t.
 * @param scale A scaling factor for each float before conversion.
 * @return An f6x32_t object storing the compressed 6-bit representation.
 */
inline __host__ __device__ f6x32_t f6_convert_sr(float32_t x, float scale = 1.0f)
{
    constexpr int seed = 1254739;
    union
    {
        float32_t float_vector;
        float float_array[32];
    } float_values{x};
    uint32_t rng =
        prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), float_values.float_array[0]);
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32(x, rng, scale);
#else
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    union
    {
        f6x32_t f6_vector;
        f6_t f6_array[32];
    } out{};

    ck::static_for<0, 32, 1>{}([&](auto i) {
        out.f6_array[i] = utils::sat_convert_to_type_sr<f6_t>(in.float_array[i] / scale, rng);
    });

    return out.f6_vector;
#endif
}

/**
 * @brief Specializes the type conversion template for converting a float into the 6-bit float type
 * (f6_t).
 *
 * Depending on the CK_USE_SR_F6_CONVERSION flag,
 * the conversion uses stochastic rounding
 * or round-to-nearest-even.
 *
 * @param x Input float value to be converted.
 * @return  The converted f6_t value.
 */
template <>
inline __host__ __device__ f6_t type_convert<f6_t, float>(float x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

/**
 * @brief Specializes the type conversion template for converting a vector of 32 floats into the
 * vector of 32 6-bit float types (f6x32_t).
 *
 * Depending on the CK_USE_SR_F6_CONVERSION flag,
 * the conversion uses stochastic rounding
 * or round-to-nearest-even.
 *
 * @param x Input float value to be converted.
 * @return  The converted f6x32_t vector.
 */
template <>
inline __host__ __device__ f6x32_t type_convert<f6x32_t, float32_t>(float32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

/**
 * @brief Specializes the type conversion template for converting the 6-bit float type (f6_t) to
 * float.
 *
 * Interprets an f6_t value as a float using the default scale factor of 1.
 *
 * @param x The 6-bit float (f6_t) value to be converted.
 * @return  The corresponding float representation.
 */
template <>
inline __host__ __device__ float type_convert<float, f6_t>(f6_t x)
{
#if defined(__gfx950__)
    union
    {
        f6x32_t f6_vector;
        f6_t f6_array[32];
    } in{x};

    union
    {
        float32_t float_vector;
        float float_array[32];
    } out{};

    out.float_vector = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(
        in.f6_vector, type_convert<float>(NumericLimits<e8m0_bexp_t>::Binary_1()));
    return out.float_array[0];
#else
    return utils::to_float<f6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x);
#endif
}

/**
 * @brief Specializes the type conversion template for converting the vector of 32 6-bit float types
 * (f6x32_t) to vector of 32 floats.
 *
 * Interprets an f6_t values as floats using the default scale factor of 1.
 *
 * @param x The vector of 32 6-bit float (f6x32_t) values to be converted.
 * @return  The corresponding float representation.
 */
template <>
inline __host__ __device__ float32_t type_convert<float32_t, f6x32_t>(f6x32_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(
        x, type_convert<float>(NumericLimits<e8m0_bexp_t>::Binary_1()));
#else
    union
    {
        f6x32_t f6_vector;
        f6_t f6_array[32];
    } in{x};

    union
    {
        float32_t float_vector;
        float float_array[32];
    } out{};

    ck::static_for<0, 32, 1>{}([&](auto i) {
        out.float_array[i] =
            utils::to_float<f6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), in.f6_array[i]);
    });

    return out.float_vector;
#endif
}

/**
 * @brief Converts a float to the 6-bit BF6 type using round-to-nearest-even.
 *
 * Divides the input by the specified scale, then saturates and converts
 * it to a 6-bit BF6 floating-point format.
 *
 * @param x     The float value to be converted.
 * @param scale The scaling factor applied to the input before conversion.
 * @return      The converted bf6_t value.
 */
inline __host__ __device__ bf6_t bf6_convert_rne(float x, float scale = 1.0f)
{
#if defined(__gfx950__)
    float16_t in1{x};
    float16_t in2{};

    union
    {
        bf6x32_t bf6_vector;
        bf6_t bf6_array[32];
    } out{};

    out.bf6_vector = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, scale);

    return out.bf6_array[0];
#else
    return utils::sat_convert_to_type<bf6_t>(x / scale);
#endif
}

/**
 * @brief Converts a vector of 32 floats to the vector of 32 6-bit BF6 types using
 * round-to-nearest-even.
 *
 * Divides the input by the specified scale, then saturates and converts
 * it to a 6-bit BF6 floating-point format.
 *
 * @param x     The float vector to be converted.
 * @param scale The scaling factor applied to the input before conversion.
 * @return      The converted bf6x32_t vector.
 */
inline __host__ __device__ bf6x32_t bf6_convert_rne(float32_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    float16_t* in1 = reinterpret_cast<float16_t*>(&x);
    float16_t* in2 = reinterpret_cast<float16_t*>(&x + 16);
    return __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(*in1, *in2, scale);
#else
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    union
    {
        bf6x32_t bf6_vector;
        bf6_t bf6_array[32];
    } out{};

    ck::static_for<0, 32, 1>{}([&](auto i) {
        out.bf6_array[i] = utils::sat_convert_to_type<bf6_t>(in.float_array[i] / scale);
    });

    return out.bf6_vector;
#endif
}

/**
 * @brief Converts a float to the 6-bit BF6 type using stochastic rounding.
 *
 * Divides the input by the specified scale,
 * and converts the result to a 6-bit BF6 floating-point
 * format with stochastic rounding.
 *
 * @param x     The float value to be converted.
 * @param scale The scaling factor applied to the input before conversion.
 * @return      The converted bf6_t value.
 */
inline __host__ __device__ bf6_t bf6_convert_sr(float x, float scale = 1.0f)
{
    constexpr int seed = 1254739;
    uint32_t rng       = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#if defined(__gfx950__)
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    union
    {
        bf6x32_t bf6_vector;
        bf6_t bf6_array[32];
    } out{};

    out.bf6_vector = __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f32(in.float_vector, rng, scale);

    return out.bf6_array[0];
#else
    return utils::sat_convert_to_type_sr<bf6_t>(x / scale, rng);
#endif
}

/**
 * @brief Converts a vector of 32 floats to the vector of 32 6-bit BF6 types using stochastic
 * rounding.
 *
 * Divides the input by the specified scale,
 * and converts the result to a 6-bit BF6 floating-point
 * format with stochastic rounding.
 *
 * @param x     The float vector to be converted.
 * @param scale The scaling factor applied to the input before conversion.
 * @return      The converted bf6x32_t vector.
 */
inline __host__ __device__ bf6x32_t bf6_convert_sr(float32_t x, float scale = 1.0f)
{
    constexpr int seed = 1254739;
    union
    {
        float32_t float_vector;
        float float_array[32];
    } float_values{x};
    uint32_t rng =
        prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), float_values.float_array[0]);
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f32(x, rng, scale);
#else
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    union
    {
        bf6x32_t bf6_vector;
        bf6_t bf6_array[32];
    } out{};

    ck::static_for<0, 32, 1>{}([&](auto i) {
        out.bf6_array[i] = utils::sat_convert_to_type_sr<bf6_t>(in.float_array[i] / scale, rng);
    });

    return out.bf6_vector;
#endif
}

/**
 * @brief Specializes float-to-bf6_t conversion.
 *
 * Uses stochastic rounding if CK_USE_SR_F6_CONVERSION is defined,
 * otherwise uses round-to-nearest-even.
 *
 * @param x Input float value to convert.
 * @return Converted bf6_t value.
 */
template <>
inline __host__ __device__ bf6_t type_convert<bf6_t, float>(float x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

/**
 * @brief Specializes vector of 32 float-to-bf6_t conversion.
 *
 * Uses stochastic rounding if CK_USE_SR_F6_CONVERSION is defined,
 * otherwise uses round-to-nearest-even.
 *
 * @param x Input float vector to convert.
 * @return Converted bf6x32_t vector.
 */
template <>
inline __host__ __device__ bf6x32_t type_convert<bf6x32_t, float32_t>(float32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

/**
 * @brief Specializes the type conversion template for converting a bf6_t value to float.
 *
 * Interprets the bf6_t value using the default scale factor of 1 and returns
 * its floating-point representation.
 *
 * @param x The bf6_t value to convert.
 * @return  The float representation of the given bf6_t value.
 */
template <>
inline __host__ __device__ float type_convert<float, bf6_t>(bf6_t x)
{
#if defined(__gfx950__)
    union
    {
        bf6x32_t bf6_vector;
        bf6_t bf6_array[32];
    } in{x};

    union
    {
        float32_t float_vector;
        float float_array[32];
    } out{};

    out.float_vector = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(
        in.bf6_vector, type_convert<float>(NumericLimits<e8m0_bexp_t>::Binary_1()));
    return out.float_array[0];
#else
    return utils::to_float<bf6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x);
#endif
}

/**
 * @brief Specializes the type conversion template for converting a vector of 32 bf6_t values to
 * vector of 32 floats.
 *
 * Interprets the bf6x32_t value using the default scale factor of 1 and returns
 * its floating-point representation.
 *
 * @param x The bf6x32_t value to convert.
 * @return  The float representation of the given vector.
 */
template <>
inline __host__ __device__ float32_t type_convert<float32_t, bf6x32_t>(bf6x32_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(
        x, type_convert<float>(NumericLimits<e8m0_bexp_t>::Binary_1()));
#else
    union
    {
        bf6x32_t bf6_vector;
        bf6_t bf6_array[32];
    } in{x};

    union
    {
        float32_t float_vector;
        float float_array[32];
    } out{};

    ck::static_for<0, 32, 1>{}([&](auto i) {
        out.float_array[i] =
            utils::to_float<bf6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), in.bf6_array[i]);
    });

    return out.float_vector;
#endif
}

#ifndef CK_CODE_GEN_RTC
template <typename Y, typename X, size_t NumElems>
inline __host__ __device__ void array_convert(std::array<Y, NumElems>& y,
                                              const std::array<X, NumElems>& x)
{
    for(size_t i = 0; i < NumElems; i++)
    {
        y[i] = type_convert<Y>(x[i]);
    }
}
#endif

template <typename Y, typename X, index_t NumElems>
inline __host__ __device__ void array_convert(Array<Y, NumElems>& y, const Array<X, NumElems>& x)
{
    for(size_t i = 0; i < NumElems; i++)
    {
        y[i] = type_convert<Y>(x[i]);
    }
}

} // namespace ck
