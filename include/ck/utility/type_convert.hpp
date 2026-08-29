// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/f8_utils.hpp"
#include "ck/utility/get_id.hpp"
#include "ck/utility/mxf4_utils.hpp"
#include "ck/utility/mxf6_utils.hpp"
#include "ck/utility/random_gen.hpp"
#include "ck/utility/array.hpp"
#include "ck/utility/amd_inline_asm.hpp"
#include "ck/utility/type.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck {
// Define the common macro for MI300 models
#if defined(__gfx942__) || defined(__gfx950__)
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
#if CK_USE_LLVM_BUILTIN_BF16 && (CK_ARCH_SUPPORT_BUILTIN_BF16 || !defined(__HIP_DEVICE_COMPILE__))
    return static_cast<__bf16>(x);
#else
    // Nan check
    if(x != x)
    {
        return bit_cast<bhalf_t>(uint16_t(0x7FC0));
    }

    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    const uint32_t first_bf16_mantisa_bit = ((u.int32 >> 16) & 1);
    constexpr uint32_t rounding_bias      = uint32_t((1 << 15) - 1);

    return bit_cast<bhalf_t>(uint16_t((u.int32 + first_bf16_mantisa_bit + rounding_bias) >> 16));
#endif
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
#if CK_USE_LLVM_BUILTIN_BF16 && (CK_ARCH_SUPPORT_BUILTIN_BF16 || !defined(__HIP_DEVICE_COMPILE__))
    return static_cast<float>(x);
#else
    union
    {
        uint32_t int32;
        float fp32;
    } u = {static_cast<uint32_t>(bit_cast<uint16_t>(x)) << 16};

    return u.fp32;
#endif
}

// convert fp32 to bfp16, round to nearest even
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, float>(float x)
{
#if CK_USE_RNE_BF16_CONVERSION
    return bf16_convert_rtn<bhalf_t>(x);
#else
    const uint32_t x_bits = bit_cast<uint32_t>(x);
    return bit_cast<bhalf_t>(uint16_t(x_bits >> 16));
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
#if CK_USE_LLVM_BUILTIN_BF16 && !defined(__HIP_DEVICE_COMPILE__)
    return static_cast<int8_t>(x);
#else
    float x_fp32 = type_convert<float>(x);

    return static_cast<int8_t>(x_fp32);
#endif
}

// convert int8 to bfp16 via fp32
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, int8_t>(int8_t x)
{
#if CK_USE_LLVM_BUILTIN_BF16 && !defined(__HIP_DEVICE_COMPILE__)
    return static_cast<bhalf_t>(x);
#else
    float x_fp32 = static_cast<float>(x);

    return type_convert<bhalf_t>(x_fp32);
#endif
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

template <typename Y, enable_if_t<is_same_v<Y, ck::tf32_t>, bool> = false>
inline __host__ __device__ constexpr float type_convert(float x)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    u.int32 = u.int32 & 0xffffe000;
    return u.fp32;
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

template <>
inline __host__ __device__ constexpr int type_convert_sp<int, f8_t>(f8_t x)
{
    union
    {
        f8_t fp8;
        int int32;
    } u = {x};

    return u.int32;
}

template <>
inline __host__ __device__ constexpr f8_t type_convert_sp<f8_t, int>(int x)
{
    union
    {
        int int32;
        f8_t fp8;
    } u = {x};

    return u.fp8;
}

template <>
inline __host__ __device__ constexpr int type_convert_sp<int, bhalf_t>(bhalf_t x)
{
    union
    {
        bhalf_t fp16;
        int int32;
    } u = {x};

    return u.int32;
}

template <>
inline __host__ __device__ constexpr bhalf_t type_convert_sp<bhalf_t, int>(int x)
{
    union
    {
        int int32;
        bhalf_t fp16;
    } u = {x};

    return u.fp16;
}

template <>
inline __host__ __device__ constexpr bhalf_t type_convert_sp<bhalf_t, float>(float x)
{
    return type_convert<bhalf_t>(x);
}

template <>
inline __host__ __device__ constexpr half_t type_convert_sp<half_t, float>(float x)
{
    return type_convert<half_t>(x);
}
// Declare a template function for fp8 conversion using SR
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_sr(X x);

// convert fp32 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_fnuz_t f8_convert_sr<f8_fnuz_t, float>(float x)
{
#if defined(__gfx950__) || defined(__gfx125__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng =
        __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() * (get_thread_global_1d_id() + 1));
#else
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x);
#endif // #ifndef CK_CODE_GEN_RTC
#endif // #if defined(__gfx950__)
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
    return f8_fnuz_t{val.i8val[0]}; // little endian
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
#if defined(__gfx950__) || defined(__gfx125__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng =
        __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() * (get_thread_global_1d_id() + 1));
#else
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x);
#endif // #ifndef CK_CODE_GEN_RTC
#endif // #if defined(__gfx950__)
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
    return bf8_fnuz_t{val.i8val[0]}; // little endian
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

/**
 * @brief Converts a float to a 8-bit float type (f8_ocp_t) using stochastic rounding.
 *
 * @param x     The input float value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_sr<f8_ocp_t, float>(float x)
{
    return f8_ocp_t{
        fp8_impl::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation, true>(
            x)};
}

/**
 * @brief Converts a vector of 2 floats to a vector of 2 8-bit float types (f8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 floats.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_sr<f8x2_ocp_t, float2_t>(float2_t x)
{
    return f8x2_ocp_t{
        fp8_impl::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation, true>(
            x)};
}

/**
 * @brief Converts a float to a 8-bit float type (bf8_ocp_t) using stochastic rounding.
 *
 * @param x     The input float value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_sr<bf8_ocp_t, float>(float x)
{
    return bf8_ocp_t{fp8_impl::cvt_float_to_fp8<bf8_ocp_t::default_interpret,
                                                bf8_ocp_t::default_saturation,
                                                true>(x)};
}

/**
 * @brief Converts a vector of 2 floats to a vector of 2 8-bit float types (bf8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 floats.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_sr<bf8x2_ocp_t, float2_t>(float2_t x)
{
    return bf8x2_ocp_t{fp8_impl::cvt_float_to_fp8<bf8_ocp_t::default_interpret,
                                                  bf8_ocp_t::default_saturation,
                                                  true>(x)};
}

/**
 * @brief Converts a half_t to a 8-bit float type (f8_ocp_t) using stochastic rounding.
 *
 * @param x     The input half_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_sr<f8_ocp_t, half_t>(half_t x)
{
    return f8_ocp_t{fp8_impl::cvt_half_t_to_fp8<f8_ocp_t::default_interpret,
                                                f8_ocp_t::default_saturation,
                                                true>(x)};
}

/**
 * @brief Converts a vector of 2 half_t to a vector of 2 8-bit float types (f8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 half_t.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_sr<f8x2_ocp_t, half2_t>(half2_t x)
{
    return f8x2_ocp_t{fp8_impl::cvt_half_t_to_fp8<f8_ocp_t::default_interpret,
                                                  f8_ocp_t::default_saturation,
                                                  true>(x)};
}

/**
 * @brief Converts a half_t to a 8-bit half_t type (bf8_ocp_t) using stochastic rounding.
 *
 * @param x     The input half_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_sr<bf8_ocp_t, half_t>(half_t x)
{
    return bf8_ocp_t{fp8_impl::cvt_half_t_to_fp8<bf8_ocp_t::default_interpret,
                                                 bf8_ocp_t::default_saturation,
                                                 true>(x)};
}

/**
 * @brief Converts a vector of 2 half_t to a vector of 2 8-bit float types (bf8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 half_t.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_sr<bf8x2_ocp_t, half2_t>(half2_t x)
{
    return bf8x2_ocp_t{fp8_impl::cvt_half_t_to_fp8<bf8_ocp_t::default_interpret,
                                                   bf8_ocp_t::default_saturation,
                                                   true>(x)};
}

/**
 * @brief Converts a bhalf_t to a 8-bit float type (f8_ocp_t) using stochastic rounding.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_sr<f8_ocp_t, bhalf_t>(bhalf_t x)
{
    return f8_ocp_t{fp8_impl::cvt_bhalf_t_to_fp8<f8_ocp_t::default_interpret,
                                                 f8_ocp_t::default_saturation,
                                                 true>(x)};
}

/**
 * @brief Converts a vector of 2 bhalf_t to a vector of 2 8-bit float types (f8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 bhalf_t.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_sr<f8x2_ocp_t, bhalf2_t>(bhalf2_t x)
{
    return f8x2_ocp_t{fp8_impl::cvt_bhalf_t_to_fp8<f8_ocp_t::default_interpret,
                                                   f8_ocp_t::default_saturation,
                                                   true>(x)};
}

/**
 * @brief Converts a bhalf_t to a 8-bit half_t type (bf8_ocp_t) using stochastic rounding.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_sr<bf8_ocp_t, bhalf_t>(bhalf_t x)
{
    return bf8_ocp_t{fp8_impl::cvt_bhalf_t_to_fp8<bf8_ocp_t::default_interpret,
                                                  bf8_ocp_t::default_saturation,
                                                  true>(x)};
}

/**
 * @brief Converts a vector of 2 bhalf_t to a vector of 2 8-bit float types (bf8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 bhalf_t.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_sr<bf8x2_ocp_t, bhalf2_t>(bhalf2_t x)
{
    return bf8x2_ocp_t{fp8_impl::cvt_bhalf_t_to_fp8<bf8_ocp_t::default_interpret,
                                                    bf8_ocp_t::default_saturation,
                                                    true>(x)};
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
    return f8_fnuz_t{val.i8val[0]};
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
    return bf8_fnuz_t{val.i8val[0]};
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

/**
 * @brief Converts a float to a 8-bit float type (f8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input float value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_rne<f8_ocp_t, float>(float x)
{
    return f8_ocp_t{
        fp8_impl::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a vector of 2 floats to a vector of 2 8-bit float types (f8_ocp_t) using rounding
 * to nearest/even.
 *
 * @param x     The input vector of 2 floats.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_rne<f8x2_ocp_t, float2_t>(float2_t x)
{
    return f8x2_ocp_t{
        fp8_impl::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a float to a 8-bit float type (bf8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input float value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_rne<bf8_ocp_t, float>(float x)
{
    return bf8_ocp_t{
        fp8_impl::cvt_float_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a vector of 2 floats to a vector of 2 8-bit float types (bf8_ocp_t) using
 * rounding to nearest/even.
 *
 * @param x     The input vector of 2 floats.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_rne<bf8x2_ocp_t, float2_t>(float2_t x)
{
    return bf8x2_ocp_t{
        fp8_impl::cvt_float_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a half_t to a 8-bit float type (f8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input half_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_rne<f8_ocp_t, half_t>(half_t x)
{
    return f8_ocp_t{
        fp8_impl::cvt_half_t_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a vector of 2 half_t to a vector of 2 8-bit float types (f8_ocp_t) using rounding
 * to nearest/even.
 *
 * @param x     The input vector of 2 half_t.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_rne<f8x2_ocp_t, half2_t>(half2_t x)
{
    return f8x2_ocp_t{
        fp8_impl::cvt_half_t_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a half_t to a 8-bit half_t type (bf8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input half_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_rne<bf8_ocp_t, half_t>(half_t x)
{
    return bf8_ocp_t{
        fp8_impl::cvt_half_t_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(
            x)};
}

/**
 * @brief Converts a vector of 2 half_t to a vector of 2 8-bit float types (bf8_ocp_t) using
 * rounding to nearest/even.
 *
 * @param x     The input vector of 2 half_t.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_rne<bf8x2_ocp_t, half2_t>(half2_t x)
{
    return bf8x2_ocp_t{
        fp8_impl::cvt_half_t_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(
            x)};
}

/**
 * @brief Converts a bhalf_t to a 8-bit float type (f8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_rne<f8_ocp_t, bhalf_t>(bhalf_t x)
{
    return f8_ocp_t{
        fp8_impl::cvt_bhalf_t_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a vector of 2 bhalf_t to a vector of 2 8-bit float types (f8_ocp_t) using
 * rounding to nearest/even.
 *
 * @param x     The input vector of 2 bhalf_t.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_rne<f8x2_ocp_t, bhalf2_t>(bhalf2_t x)
{
    return f8x2_ocp_t{
        fp8_impl::cvt_bhalf_t_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a bhalf_t to a 8-bit half_t type (bf8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_rne<bf8_ocp_t, bhalf_t>(bhalf_t x)
{
    return bf8_ocp_t{
        fp8_impl::cvt_bhalf_t_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(
            x)};
}

/**
 * @brief Converts a vector of 2 bhalf_t to a vector of 2 8-bit float types (bf8_ocp_t) using
 * rounding to nearest/even.
 *
 * @param x     The input vector of 2 bhalf_t.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_rne<bf8x2_ocp_t, bhalf2_t>(bhalf2_t x)
{
    return bf8x2_ocp_t{
        fp8_impl::cvt_bhalf_t_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(
            x)};
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

// convert fp8 to fp32
template <>
inline __host__ __device__ float type_convert<float, f8_fnuz_t>(f8_fnuz_t x)
{
#if defined(__gfx94__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(static_cast<uint8_t>(x));
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

/**
 * @brief Converts a f8_ocp_t value to a float value.
 *
 * @param x     The input f8_ocp_t value.
 * @return      The converted float value.
 */
template <>
inline __host__ __device__ float type_convert<float, f8_ocp_t>(f8_ocp_t x)
{
#if CK_OCP_FP8_CVT_FAST_PATH
    union
    {
        unsigned int i32val;
        fp8_storage_t i8val[4];
    } val;
    val.i8val[0] = x.data;
    return __builtin_amdgcn_cvt_f32_fp8(val.i32val, 0);
#else
    return fp8_impl::cast_from_f8<float, f8_ocp_t::wm, f8_ocp_t::we, false>(x.data);
#endif
}

/**
 * @brief Converts a vector of 2 f8_ocp_t values to a vector of 2 float values.
 *
 * @param x     The input vector of 2 f8_ocp_t values.
 * @return      The converted vector of 2 float values.
 */
template <>
inline __host__ __device__ float2_t type_convert<float2_t, f8x2_ocp_t>(f8x2_ocp_t x)
{
#if CK_OCP_FP8_CVT_FAST_PATH
// __builtin_amdgcn_cvt_pk_f32_fp8 can produce incorrect results due to a compiler issue.
// TODO: Enable when SWDEV-532959 is fixed.
#if defined(__gfx120__)
    return float2_t{__builtin_amdgcn_cvt_f32_fp8(bit_cast<uint16_t>(x), 0),
                    __builtin_amdgcn_cvt_f32_fp8(bit_cast<uint16_t>(x), 1)};
#else
    return __builtin_amdgcn_cvt_pk_f32_fp8(bit_cast<uint16_t>(x), false);
#endif
#else
    return float2_t{fp8_impl::cast_from_f8<float, f8_ocp_t::wm, f8_ocp_t::we, false>(
                        x.AsType<fp8_storage_t>()[Number<0>{}]),
                    fp8_impl::cast_from_f8<float, f8_ocp_t::wm, f8_ocp_t::we, false>(
                        x.AsType<fp8_storage_t>()[Number<1>{}])};
#endif
}

/**
 * @brief Converts a f8_ocp_t value to a half_t value.
 *
 * @param x     The input f8_ocp_t value.
 * @return      The converted half_t value.
 */
template <>
inline __host__ __device__ half_t type_convert<half_t, f8_ocp_t>(f8_ocp_t x)
{
#if defined(__gfx950__)
    union
    {
        uint16_t i16val;
        fp8_storage_t i8val[2];
    } input;
    input.i8val[0] = x.data;

    union
    {
        half2_t half_vec;
        half_t half_arr[2];
    } output;
    output.half_vec = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(input.i16val, /*scale*/ 1.f, 0);

    return output.half_arr[0];
#elif defined(__gfx125__)
    union
    {
        int ival;
        fp8_storage_t i8val[4];
    } input{};
    input.i8val[0] = x.data;
    return __builtin_amdgcn_cvt_f16_fp8(input.ival, 0);
#else
    return fp8_impl::cast_from_f8<half_t, f8_ocp_t::wm, f8_ocp_t::we, false>(x.data);
#endif
}

/**
 * @brief Converts a vector of 2 f8_ocp_t values to a vector of 2 half_t values.
 *
 * @param x     The input vector of 2 f8_ocp_t values.
 * @return      The converted vector of 2 half_t values.
 */
template <>
inline __host__ __device__ half2_t type_convert<half2_t, f8x2_ocp_t>(f8x2_ocp_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(bit_cast<uint16_t>(x), /*scale*/ 1.f, 0);
#elif defined(__gfx125__)
    return __builtin_amdgcn_cvt_pk_f16_fp8(bit_cast<uint16_t>(x));
#else
    return half2_t{type_convert<half_t>(float(x.AsType<f8_ocp_t>()[Number<0>{}])),
                   type_convert<half_t>(float(x.AsType<f8_ocp_t>()[Number<1>{}]))};
#endif
}

/**
 * @brief Converts a f8_ocp_t value to a bhalf_t value.
 *
 * @param x     The input f8_ocp_t value.
 * @return      The converted bhalf_t value.
 */
template <>
inline __host__ __device__ bhalf_t type_convert<bhalf_t, f8_ocp_t>(f8_ocp_t x)
{
#if defined(__gfx950__)
    union
    {
        uint16_t i16val;
        fp8_storage_t i8val[2];
    } input;
    input.i8val[0] = x.data;

    union
    {
        bhalf2_t bhalf_vec;
        bhalf_t bhalf_arr[2];
    } output;
    output.bhalf_vec = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(input.i16val, /*scale*/ 1.f, 0);

    return output.bhalf_arr[0];
#elif defined(__gfx125__)
    union
    {
        fp8_impl::uint32x2_t ival;
        f8_ocp_t f8val[8];
    } input{};
    input.f8val[0] = x;
    union
    {
        bhalf8_t bhalf_vec;
        bhalf_t bhalf_arr[8];
    } output;
    output.bhalf_vec = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(input.ival, uint32_t{0x7F}, 0);
    return output.bhalf_arr[0];
#else
    return type_convert<bhalf_t>(
        fp8_impl::cast_from_f8<float, f8_ocp_t::wm, f8_ocp_t::we, false>(x.data));
#endif
}

/**
 * @brief Converts a vector of 2 f8_ocp_t values to a vector of 2 bhalf_t values.
 *
 * @param x     The input vector of 2 f8_ocp_t values.
 * @return      The converted vector of 2 bhalf_t values.
 */
template <>
inline __host__ __device__ bhalf2_t type_convert<bhalf2_t, f8x2_ocp_t>(f8x2_ocp_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(bit_cast<uint16_t>(x), /*scale*/ 1.f, 0);
#elif defined(__gfx125__)
    union
    {
        fp8_impl::uint32x2_t ival;
        f8x2_ocp_t f8x2val[4];
    } input{};
    input.f8x2val[0] = x;
    union
    {
        bhalf8_t bhalf_vec;
        bhalf2_t bhalf_arr[4];
    } output;
    output.bhalf_vec = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(input.ival, uint32_t{0x7F}, 0);
    return output.bhalf_arr[0];
#else
    return bhalf2_t{type_convert<bhalf_t>(float(x.AsType<f8_ocp_t>()[Number<0>{}])),
                    type_convert<bhalf_t>(float(x.AsType<f8_ocp_t>()[Number<1>{}]))};
#endif
}

/**
 * @brief Converts a bf8_ocp_t value to a float value.
 *
 * @param x     The input bf8_ocp_t value.
 * @return      The converted float value.
 */
template <>
inline __host__ __device__ float type_convert<float, bf8_ocp_t>(bf8_ocp_t x)
{
#if CK_OCP_FP8_CVT_FAST_PATH
    union
    {
        unsigned int i32val;
        fp8_storage_t i8val[4];
    } val;
    val.i8val[0] = x.data;
    return __builtin_amdgcn_cvt_f32_bf8(val.i32val, 0);
#else
    return fp8_impl::cast_from_f8<float, bf8_ocp_t::wm, bf8_ocp_t::we, false>(x.data);
#endif
}

/**
 * @brief Converts a vector of 2 bf8_ocp_t values to a vector of 2 float values.
 *
 * @param x     The input vector of 2 bf8_ocp_t values.
 * @return      The converted vector of 2 float values.
 */
template <>
inline __host__ __device__ float2_t type_convert<float2_t, bf8x2_ocp_t>(bf8x2_ocp_t x)
{
#if CK_OCP_FP8_CVT_FAST_PATH
// __builtin_amdgcn_cvt_pk_f32_bf8 can produce incorrect results due to a compiler issue.
// TODO: Enable when SWDEV-532959 is fixed.
#if defined(__gfx120__)
    return float2_t{__builtin_amdgcn_cvt_f32_bf8(bit_cast<uint16_t>(x), 0),
                    __builtin_amdgcn_cvt_f32_bf8(bit_cast<uint16_t>(x), 1)};
#else
    return __builtin_amdgcn_cvt_pk_f32_bf8(bit_cast<uint16_t>(x), false);
#endif
#else
    return float2_t{fp8_impl::cast_from_f8<float, bf8_ocp_t::wm, bf8_ocp_t::we, false>(
                        x.AsType<fp8_storage_t>()[Number<0>{}]),
                    fp8_impl::cast_from_f8<float, bf8_ocp_t::wm, bf8_ocp_t::we, false>(
                        x.AsType<fp8_storage_t>()[Number<1>{}])};
#endif
}

/**
 * @brief Converts a bf8_ocp_t value to a half_t value.
 *
 * @param x     The input bf8_ocp_t value.
 * @return      The converted half_t value.
 */
template <>
inline __host__ __device__ half_t type_convert<half_t, bf8_ocp_t>(bf8_ocp_t x)
{
#if defined(__gfx950__)
    union
    {
        uint16_t i16val;
        fp8_storage_t i8val[2];
    } val;
    val.i8val[0] = x.data;
    return __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(val.i16val, /*scale*/ 1.f, 0)[0];
#elif defined(__gfx125__)
    union
    {
        int ival;
        fp8_storage_t i8val[4];
    } input{};
    input.i8val[0] = x.data;
    return __builtin_amdgcn_cvt_f16_bf8(input.ival, 0);
#else
    return fp8_impl::cast_from_f8<half_t, bf8_ocp_t::wm, bf8_ocp_t::we, false>(x.data);
#endif
}

/**
 * @brief Converts a vector of 2 bf8_ocp_t values to a vector of 2 half_t values.
 *
 * @param x     The input vector of 2 bf8_ocp_t values.
 * @return      The converted vector of 2 half_t values.
 */
template <>
inline __host__ __device__ half2_t type_convert<half2_t, bf8x2_ocp_t>(bf8x2_ocp_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(bit_cast<uint16_t>(x), /*scale*/ 1.f, 0);
#elif defined(__gfx125__)
    return __builtin_amdgcn_cvt_pk_f16_bf8(bit_cast<uint16_t>(x));
#else
    return half2_t{type_convert<half_t>(float(x.AsType<bf8_ocp_t>()[Number<0>{}])),
                   type_convert<half_t>(float(x.AsType<bf8_ocp_t>()[Number<1>{}]))};
#endif
}

/**
 * @brief Converts a bf8_ocp_t value to a bhalf_t value.
 *
 * @param x     The input bf8_ocp_t value.
 * @return      The converted bhalf_t value.
 */
template <>
inline __host__ __device__ bhalf_t type_convert<bhalf_t, bf8_ocp_t>(bf8_ocp_t x)
{
#if defined(__gfx950__)
    union
    {
        uint16_t i16val;
        fp8_storage_t i8val[2];
    } input;
    input.i8val[0] = x.data;

    union
    {
        bhalf2_t bhalf_vec;
        bhalf_t bhalf_arr[2];
    } output;
    output.bhalf_vec = __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(input.i16val, /*scale*/ 1.f, 0);

    return output.bhalf_arr[0];
#elif defined(__gfx125__)
    union
    {
        fp8_impl::uint32x2_t ival;
        bf8_ocp_t bf8val[8];
    } input{};
    input.bf8val[0] = x;
    union
    {
        bhalf8_t bhalf_vec;
        bhalf_t bhalf_arr[8];
    } output;
    output.bhalf_vec = __builtin_amdgcn_cvt_scale_pk8_bf16_bf8(input.ival, uint32_t{0x7F}, 0);
    return output.bhalf_arr[0];
#else
    return type_convert<bhalf_t>(
        fp8_impl::cast_from_f8<float, bf8_ocp_t::wm, bf8_ocp_t::we, false>(x.data));
#endif
}

/**
 * @brief Converts a vector of 2 bf8_ocp_t values to a vector of 2 bhalf_t values.
 *
 * @param x     The input vector of 2 bf8_ocp_t values.
 * @return      The converted vector of 2 bhalf_t values.
 */
template <>
inline __host__ __device__ bhalf2_t type_convert<bhalf2_t, bf8x2_ocp_t>(bf8x2_ocp_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(bit_cast<uint16_t>(x), /*scale*/ 1.f, 0);
#elif defined(__gfx125__)
    union
    {
        fp8_impl::uint32x2_t ival;
        bf8x2_ocp_t f8x2val[4];
    } input{};
    input.f8x2val[0] = x;
    union
    {
        bhalf8_t bhalf_vec;
        bhalf2_t bhalf_arr[4];
    } output;
    output.bhalf_vec = __builtin_amdgcn_cvt_scale_pk8_bf16_bf8(input.ival, uint32_t{0x7F}, 0);
    return output.bhalf_arr[0];
#else
    return bhalf2_t{type_convert<bhalf_t>(float(x.AsType<bf8_ocp_t>()[Number<0>{}])),
                    type_convert<bhalf_t>(float(x.AsType<bf8_ocp_t>()[Number<1>{}]))};
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
#else
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

/**
 * @brief Converts a half_t value to a f8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input half_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t type_convert<f8_ocp_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_ocp_t>(x);
#else
    return f8_convert_rne<f8_ocp_t>(x);
#endif
}

/**
 * @brief Converts a half_t value to a bf8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input half_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t type_convert<bf8_ocp_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_ocp_t>(x);
#else
    return f8_convert_rne<bf8_ocp_t>(x);
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

/**
 * @brief Converts a float value to a f8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input float value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t type_convert<f8_ocp_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_ocp_t>(x);
#else
    return f8_convert_rne<f8_ocp_t>(x);
#endif
}

/**
 * @brief Converts a float value to a bf8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input float value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t type_convert<bf8_ocp_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_ocp_t>(x);
#else
    return f8_convert_rne<bf8_ocp_t>(x);
#endif
}

/**
 * @brief Converts a bhalf_t value to a f8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t type_convert<f8_ocp_t, bhalf_t>(bhalf_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_ocp_t>(x);
#else
    return f8_convert_rne<f8_ocp_t>(x);
#endif
}

/**
 * @brief Converts a bhalf_t value to a bf8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t type_convert<bf8_ocp_t, bhalf_t>(bhalf_t x)
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
    uint32_t i32val = static_cast<uint32_t>(static_cast<uint8_t>(x));
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

#ifndef CK_CODE_GEN_RTC
// convert fp32 to fp4 with round to nearest even
template <typename T, typename = enable_if_t<scalar_type<T>::vector_size == 1>>
inline __host__ __device__ f4_t f4_convert_rne(T x, float scale = 1.f)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_to_f4_scaled<T, false>(x, scale);
#else
    float f = type_convert<float>(x);
    return utils::sat_convert_to_type<f4_t>(f / scale);
#endif
}

template <typename T, typename = enable_if_t<scalar_type<T>::vector_size == 2>>
inline __host__ __device__ f4x2_t f4_convert_rne(T x, float scale = 1.f)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_to_f4_scaled<T, false>(x, scale);
#else
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};
    float f0      = type_convert<float>(x[0]);
    float f1      = type_convert<float>(x[1]);
    uint8_t l     = utils::sat_convert_to_type<f4_t>(f0 / scale);
    uint8_t h     = utils::sat_convert_to_type<f4_t>(f1 / scale);
    value.bitwise = (h << 4) | l;
    return value.f4x2_array[0];
#endif
}

template <typename T, typename = enable_if_t<scalar_type<T>::vector_size == 8>>
inline __host__ __device__ f4x8_t f4_convert_rne(T x, float scale = 1.f)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_to_f4_scaled<T, false>(x, scale);
#else
    union
    {
        uint8_t i8x4[4];
        f4x8_t f4x8;
    } ret{};

    ck::static_for<0, 4, 1>{}([&](auto i) {
        float f0    = type_convert<float>(x[static_cast<int>(2 * i)]);
        float f1    = type_convert<float>(x[static_cast<int>(2 * i + 1)]);
        uint8_t l   = utils::sat_convert_to_type<f4_t>(f0 / scale);
        uint8_t h   = utils::sat_convert_to_type<f4_t>(f1 / scale);
        ret.i8x4[i] = (h << 4) | l;
    });
    return ret.f4x8;
#endif
}

template <typename T, typename = enable_if_t<scalar_type<T>::vector_size == 32>>
inline __host__ __device__ f4x32_t f4_convert_rne(T x, float scale = 1.f)
{
    using BaseT         = typename scalar_type<T>::type;
    constexpr int Npack = scalar_type<T>::vector_size / 8;

    union
    {
        f4x32_t v32f4;
        f4x8_t v8f4x4[Npack];
    } ret{};
    union
    {
        T v32;
        typename vector_type<BaseT, 8>::type v8_array[Npack];
    } value{x};

    ck::static_for<0, Npack, 1>{}(
        [&](auto idx) { ret.v8f4x4[Number<idx>{}] = f4_convert_rne(value.v8_array[idx], scale); });

    return ret.v32f4;
}

// convert fp32 to fp4 with stochastic rounding
template <typename T, typename = enable_if_t<scalar_type<T>::vector_size == 1>>
inline __host__ __device__ f4_t f4_convert_sr(T x, float scale = 1.f)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_to_f4_scaled<T, true>(x, scale);
#else
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x);
#endif // #ifndef CK_CODE_GEN_RTC
    float f = type_convert<float>(x);
    return utils::sat_convert_to_type_sr<f4_t>(f / scale, rng);
#endif
}

template <typename T, typename = enable_if_t<scalar_type<T>::vector_size == 2>>
inline __host__ __device__ f4x2_t f4_convert_sr(T x, float scale = 1.f)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_to_f4_scaled<T, true>(x, scale);
#else
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x[0]);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x[0]);
#endif // #ifndef CK_CODE_GEN_RTC

    float f0      = type_convert<float>(x[0]);
    float f1      = type_convert<float>(x[1]);
    uint8_t l     = utils::sat_convert_to_type_sr<f4_t>(f0 / scale, rng);
    uint8_t h     = utils::sat_convert_to_type_sr<f4_t>(f1 / scale, rng);
    value.bitwise = (h << 4) | l;
    return value.f4x2_array[0];
#endif
}

template <typename T, typename = enable_if_t<scalar_type<T>::vector_size == 8>>
inline __host__ __device__ f4x8_t f4_convert_sr(T x, float scale = 1.f)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_to_f4_scaled<T, false>(x, scale);
#else
    union
    {
        uint8_t i8x4[4];
        f4x8_t f4x8;
    } ret{};

    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x[0]);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x[0]);
#endif // #ifndef CK_CODE_GEN_RTC

    ck::static_for<0, 4, 1>{}([&](auto i) {
        float f0    = type_convert<float>(x[static_cast<int>(2 * i)]);
        float f1    = type_convert<float>(x[static_cast<int>(2 * i + 1)]);
        uint8_t l   = utils::sat_convert_to_type_sr<f4_t>(f0 / scale, rng);
        uint8_t h   = utils::sat_convert_to_type_sr<f4_t>(f1 / scale, rng);
        ret.i8x4[i] = (h << 4) | l;
    });
    return ret.f4x8;
#endif
}

template <typename T, typename = enable_if_t<scalar_type<T>::vector_size == 32>>
inline __host__ __device__ f4x32_t f4_convert_sr(T x, float scale = 1.f)
{
    using BaseT         = typename scalar_type<T>::type;
    constexpr int Npack = scalar_type<T>::vector_size / 8;

    union
    {
        f4x32_t v32f4;
        f4x8_t v8f4x4[Npack];
    } ret{};
    union
    {
        T v32;
        typename vector_type<BaseT, 8>::type v8_array[Npack];
    } value{x};

    ck::static_for<0, Npack, 1>{}(
        [&](auto idx) { ret.v8f4x4[Number<idx>{}] = f4_convert_sr(value.v8_array[idx], scale); });

    return ret.v32f4;
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
template <>
inline __host__ __device__ f4x2_pk_t type_convert<f4x2_pk_t, float2_t>(float2_t x)
{
    return static_cast<f4x2_pk_t>(type_convert<f4x2_t>(x));
}

// convert vector of 8 fp32 to vector of 8 fp4
template <>
inline __host__ __device__ f4x8_t type_convert<f4x8_t, float8_t>(float8_t x)
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
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<float>(x);
#else
    return utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x);
#endif
}

// convert vector of 2 fp4 to vector of 2 fp32
template <>
inline __host__ __device__ float2_t type_convert<float2_t, f4x2_t>(f4x2_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<float2_t>(x);
#else
    float2_t ret{
        utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(),
                              x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{})),
        utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(),
                              x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}))};
    return ret;
#endif
}

// convert vector of 8 fp4 to vector of 8 fp32
template <>
inline __host__ __device__ float8_t type_convert<float8_t, f4x8_t>(f4x8_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<float8_t>(x);
#else
    union
    {
        float8_t vf32_8x1;
        float vf32[8];
    } ret{};

    ck::static_for<0, 4, 1>{}([&](auto i) {
        ret.vf32[2 * i] = utils::to_float<f4_t>(
            NumericLimits<e8m0_bexp_t>::Binary_1(),
            x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<0>{}));
        ret.vf32[2 * i + 1] = utils::to_float<f4_t>(
            NumericLimits<e8m0_bexp_t>::Binary_1(),
            x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<1>{}));
    });
    return ret.vf32_8x1;
#endif
}

// convert vector of 32 fp4 to vector of 32 fp32
template <>
inline __host__ __device__ float32_t type_convert<float32_t, f4x32_t>(f4x32_t x)
{
    constexpr int N = 32 / 8;
    union
    {
        f4x32_t vf4;
        f4x8_t v8f4[N];
    } value{x};
    union
    {
        float32_t vf32;
        float8_t v8f32[N];
    } ret{};

    ck::static_for<0, N, 1>{}(
        [&](auto idx) { ret.v8f32[idx] = type_convert<float8_t>(value.v8f4[idx]); });
    return ret.vf32;
}

// convert f16 to f4
template <>
inline __host__ __device__ f4_t type_convert<f4_t, half_t>(half_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ f4x2_t type_convert<f4x2_t, half2_t>(half2_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}
template <>
inline __host__ __device__ f4x2_pk_t type_convert<f4x2_pk_t, half2_t>(half2_t x)
{
    return static_cast<f4x2_pk_t>(type_convert<f4x2_t>(x));
}

// convert vector of 8 fp32 to vector of 8 fp4
template <>
inline __host__ __device__ f4x8_t type_convert<f4x8_t, half8_t>(half8_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

// convert vector of 32 fp32 to vector of 32 fp4
template <>
inline __host__ __device__ f4x32_t type_convert<f4x32_t, half32_t>(half32_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

// convert fp4 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, f4_t>(f4_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<half_t>(x);
#else
    return type_convert<half_t>(utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x));
#endif
}

template <>
inline __host__ __device__ half2_t type_convert<half2_t, f4x2_t>(f4x2_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<half2_t>(x);
#else
    half2_t ret{type_convert<half_t>(utils::to_float<f4_t>(
                    NumericLimits<e8m0_bexp_t>::Binary_1(),
                    x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}))),
                type_convert<half_t>(utils::to_float<f4_t>(
                    NumericLimits<e8m0_bexp_t>::Binary_1(),
                    x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{})))};
    return ret;
#endif
}

template <>
inline __host__ __device__ half8_t type_convert<half8_t, f4x8_t>(f4x8_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<half8_t>(x);
#else
    union
    {
        half8_t vf16_8x1;
        half_t vf16[8];
    } ret{};

    ck::static_for<0, 4, 1>{}([&](auto i) {
        ret.vf16[2 * i]     = type_convert<half_t>(utils::to_float<f4_t>(
            NumericLimits<e8m0_bexp_t>::Binary_1(),
            x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<0>{})));
        ret.vf16[2 * i + 1] = type_convert<half_t>(utils::to_float<f4_t>(
            NumericLimits<e8m0_bexp_t>::Binary_1(),
            x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<1>{})));
    });
    return ret.vf16_8x1;
#endif
}

template <>
inline __host__ __device__ half32_t type_convert<half32_t, f4x32_t>(f4x32_t x)
{
    constexpr int N = 32 / 8;
    union
    {
        f4x32_t vf4;
        f4x8_t v8f4[N];
    } value{x};
    union
    {
        half32_t vf16;
        half8_t v8f16[N];
    } ret{};

    ck::static_for<0, N, 1>{}(
        [&](auto idx) { ret.v8f16[idx] = type_convert<half8_t>(value.v8f4[idx]); });
    return ret.vf16;
}

// convert bf16 to f4
template <>
inline __host__ __device__ f4_t type_convert<f4_t, bhalf_t>(bhalf_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ f4x2_t type_convert<f4x2_t, bhalf2_t>(bhalf2_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}
template <>
inline __host__ __device__ f4x2_pk_t type_convert<f4x2_pk_t, bhalf2_t>(bhalf2_t x)
{
    return static_cast<f4x2_pk_t>(type_convert<f4x2_t>(x));
}

// convert vector of 8 fp32 to vector of 8 fp4
template <>
inline __host__ __device__ f4x8_t type_convert<f4x8_t, bhalf8_t>(bhalf8_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

// convert vector of 32 fp32 to vector of 32 fp4
template <>
inline __host__ __device__ f4x32_t type_convert<f4x32_t, bhalf32_t>(bhalf32_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

// convert fp4 to bf16
template <>
inline __host__ __device__ bhalf_t type_convert<bhalf_t, f4_t>(f4_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<bhalf_t>(x);
#else
    return type_convert<bhalf_t>(utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x));
#endif
}

template <>
inline __host__ __device__ bhalf2_t type_convert<bhalf2_t, f4x2_t>(f4x2_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<bhalf2_t>(x);
#else
    bhalf2_t ret{type_convert<bhalf_t>(utils::to_float<f4_t>(
                     NumericLimits<e8m0_bexp_t>::Binary_1(),
                     x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}))),
                 type_convert<bhalf_t>(utils::to_float<f4_t>(
                     NumericLimits<e8m0_bexp_t>::Binary_1(),
                     x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{})))};
    return ret;
#endif
}

template <>
inline __host__ __device__ bhalf8_t type_convert<bhalf8_t, f4x8_t>(f4x8_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<bhalf8_t>(x);
#else
    union
    {
        bhalf8_t vf16_8x1;
        bhalf_t vf16[8];
    } ret{};

    ck::static_for<0, 4, 1>{}([&](auto i) {
        ret.vf16[2 * i]     = type_convert<bhalf_t>(utils::to_float<f4_t>(
            NumericLimits<e8m0_bexp_t>::Binary_1(),
            x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<0>{})));
        ret.vf16[2 * i + 1] = type_convert<bhalf_t>(utils::to_float<f4_t>(
            NumericLimits<e8m0_bexp_t>::Binary_1(),
            x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<1>{})));
    });
    return ret.vf16_8x1;
#endif
}

template <>
inline __host__ __device__ bhalf32_t type_convert<bhalf32_t, f4x32_t>(f4x32_t x)
{
    constexpr int N = 32 / 8;
    union
    {
        f4x32_t vf4;
        f4x8_t v8f4[N];
    } value{x};
    union
    {
        bhalf32_t vf16;
        bhalf8_t v8f16[N];
    } ret{};

    ck::static_for<0, N, 1>{}(
        [&](auto idx) { ret.v8f16[idx] = type_convert<bhalf8_t>(value.v8f4[idx]); });
    return ret.vf16;
}
/**
 * @brief Fallback conversion to 6-bit floating-point using round-to-nearest-even.
 *
 * This function provides implementation for converting to 6-bit floating-point
 * formats when hardware intrinsics are unavailable (CK_MX_FP6_CVT_FAST_PATH = 0). It supports
 * scalar and vector types with round-to-nearest-even (RNE) rounding and saturation for out-of-range
 * values.
 *
 * @tparam T_F6 Target 6-bit floating-point type (f6_t, bf6_t, or their vector types).
 * @tparam T    Source data type (scalar or vector).
 * @param x     The value to convert.
 * @param scale Scaling factor applied before conversion (default: 1.0).
 * @return      The converted value in 6-bit floating-point format.
 *
 * @note Used when fast hardware conversion instructions are not available.
 * @note Values are first converted to float, divided by scale, then saturated to f6 range.
 */
template <typename T_F6, typename T>
inline __host__ __device__ T_F6 slowcast_to_f6_rne(T x, float scale = 1.0f)
{
    constexpr int N = scalar_type<T>::vector_size;
    if constexpr(N == 1)
    {
        return utils::sat_convert_to_type<T_F6>(type_convert<float>(x) / scale);
    }
    else
    {
        using BaseT    = typename scalar_type<T>::type;
        using BaseT_F6 = typename utils::get_f6_bit_type<T_F6>::type;
        union
        {
            T vector;
            BaseT array[N];
        } in{x};

        using array_type = NativeVectorT<uint8_t, N>;
        array_type uint8_array;

        // collect the 6-bit values into an array
        static_for<0, N, 1>{}([&](auto i) {
            uint8_array[static_cast<index_t>(i)] =
                utils::sat_convert_to_type<BaseT_F6>(type_convert<float>(in.array[i]) / scale);
        });
        return T_F6{f6_pk_t<BaseT_F6, N>{uint8_array}};
    }
}

/**
 * @brief Fallback conversion to 6-bit floating-point using stochastic rounding.
 *
 * This function provides implementation for converting to 6-bit floating-point
 * formats when hardware intrinsics are unavailable (CK_MX_FP6_CVT_FAST_PATH = 0). It supports
 * scalar and vector types with stochastic rounding (SR) and saturation for out-of-range
 * values.
 *
 * @tparam T_F6 Target 6-bit floating-point type (f6_t, bf6_t, or their vector types).
 * @tparam T    Source data type (scalar or vector).
 * @param x     The value to convert.
 * @param scale Scaling factor applied before conversion (default: 1.0).
 * @return      The converted value in 6-bit floating-point format.
 *
 * @note Used when fast hardware conversion instructions are not available.
 * @note Values are first converted to float, divided by scale, then saturated to f6 range.
 */
template <typename T_F6, typename T>
inline __host__ __device__ T_F6 slowcast_to_f6_sr(T x, float scale = 1.0f)
{
    using BaseT     = typename scalar_type<T>::type;
    constexpr int N = scalar_type<T>::vector_size;
    union
    {
        T vector;
        BaseT array[N];
    } in{x};

    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), in.array[0]);
#else  // #ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), in.array[0]);
#endif // #ifndef CK_CODE_GEN_RTC

    if constexpr(N == 1)
    {
        return utils::sat_convert_to_type_sr<T_F6>(type_convert<float>(x) / scale, rng);
    }
    else
    {
        using T_F6_PK  = utils::get_f6_packed_type_t<T_F6>;
        using BaseT_F6 = typename utils::get_f6_bit_type<T_F6>::type; // fp6_t or bf6_t
        T_F6_PK out{};

        static_for<0, N, 1>{}([&](auto i) {
            auto f6_val = utils::sat_convert_to_type_sr<BaseT_F6>(
                type_convert<float>(in.array[i]) / scale, rng);
            out.pack(f6_val, static_cast<int>(i));
        });

        return T_F6{out};
    }
}

/**
 * @brief Fallback conversion from 6-bit floating-point to higher precision with E8M0 scaling.
 *
 * This function provides implementation for converting from 6-bit floating-point
 * formats when hardware intrinsics are unavailable (CK_MX_FP6_CVT_FAST_PATH = 0).
 *
 * @tparam T      Target data type (scalar or vector of float, half_t, or bhalf_t).
 * @tparam T_F6   Source 6-bit floating-point type (f6_t, bf6_t, or their vector types).
 * @param x       The 6-bit floating-point value to convert.
 * @param scale   E8M0 exponent scale factor applied during conversion (default: 2^0 = 1.0).
 * @return        The converted value in the target precision format.
 */
template <typename T, typename T_F6>
inline __host__ __device__ T
slowcast_from_f6(T_F6 x, e8m0_bexp_t scale = NumericLimits<e8m0_bexp_t>::Binary_1())
{
    constexpr int N = scalar_type<T>::vector_size;
    using BaseT     = typename scalar_type<T>::type;
    if constexpr(N == 1)
    {
        return type_convert<BaseT>(utils::to_float<T_F6>(scale, x));
    }
    else
    {
        using T_F6_PK    = utils::get_f6_packed_type_t<T_F6>;
        T_F6_PK x_packed = static_cast<T_F6_PK>(x);

        union
        {
            T vector;
            BaseT array[N];
        } out{};

        static_for<0, N, 1>{}([&](auto i) {
            auto f6_elem = x_packed.unpack(i);
            out.array[i] = type_convert<BaseT>(utils::to_float<decltype(f6_elem)>(scale, f6_elem));
        });

        return out.vector;
    }
}

/**
 * @brief Converts a scalar or vector of float/float16/bfloat16 to packed 6-bit representation.
 *
 * Divides each input element by the provided scale value, then performs conversion
 * with rounding to nearest even to pack each element into 6 bits of precision.
 *
 * @tparam X Input type (scalar or vector of float/float16/bfloat16 with size 1, 16, or 32)
 * @param x     Input scalar or vector value
 * @param scale Scaling factor applied before conversion
 * @return      Converted Y type storing the compressed 6-bit representation (bf6_t, bf6x16_t,
 * bf6x32_t)
 */
template <typename X>
inline __host__ __device__ auto f6_convert_rne(X x, float scale = 1.0f) ->
    typename utils::f6_result_type<X>::type
{
    /* derived return type */
    using Y = typename utils::f6_result_type<X>::type;

    /* template constraints */
    constexpr int N = scalar_type<X>::vector_size;
    using BaseT     = typename scalar_type<X>::type;
    static_assert(N == 1 || N == 16 || N == 32, "Unsupported vector size");
    static_assert(is_same_v<BaseT, float> || is_same_v<BaseT, half_t> || is_same_v<BaseT, bhalf_t>,
                  "Input type must be float, half_t, or bhalf_t");

#if CK_MX_FP6_CVT_FAST_PATH
    return cast_to_f6_scaled<Y, false>(x, scale);
#else
    return slowcast_to_f6_rne<Y>(x, scale);
#endif
}

/**
 * @brief Converts a scalar or vector of float/half_t/bhalf_t to packed 6-bit representation.
 *
 * Divides each input element by the provided scale value, then performs conversion
 * stochastic rounding to pack each element into 6 bits of precision.
 *
 * @tparam X Input type (scalar or vector of float/float16/bfloat16 with size 1, 16, or 32)
 * @param x     Input scalar or vector value
 * @param scale Scaling factor applied before conversion
 * @return      Converted Y type storing the compressed 6-bit representation (bf6_t, bf6x16_t,
 * bf6x32_t)
 */
template <typename X>
inline __host__ __device__ auto f6_convert_sr(X x, float scale = 1.0f) ->
    typename utils::f6_result_type<X>::type
{
    /* derived return type */
    using Y = typename utils::f6_result_type<X>::type;

    /* template constraints */
    constexpr int N = scalar_type<X>::vector_size;
    using BaseT     = typename scalar_type<X>::type;
    static_assert(N == 1 || N == 16 || N == 32, "Unsupported vector size");
    static_assert(is_same_v<BaseT, float> || is_same_v<BaseT, half_t> || is_same_v<BaseT, bhalf_t>,
                  "Input type must be float, half_t, or bhalf_t");

#if CK_MX_FP6_CVT_FAST_PATH
    return cast_to_f6_scaled<Y, true>(x, scale);
#else
    return slowcast_to_f6_sr<Y>(x, scale);
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

template <>
inline __host__ __device__ f6x32_pk_t type_convert<f6x32_pk_t, float32_t>(float32_t x)
{
    return static_cast<f6x32_pk_t>(type_convert<f6x32_t>(x));
}

template <>
inline __host__ __device__ f6x16_t type_convert<f6x16_t, float16_t>(float16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ f6x16_pk_t type_convert<f6x16_pk_t, float16_t>(float16_t x)
{
    return static_cast<f6x16_pk_t>(type_convert<f6x16_t>(x));
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
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float>(x, 1.0f);
#else
    return slowcast_from_f6<float>(x);
#endif
}

/**
 * @brief Specializes the type conversion template for converting the vector of 32 6-bit float
 * types (f6x32_t) to vector of 32 floats.
 *
 * Interprets an f6_t values as floats using the default scale factor of 1.
 *
 * @param x The vector of 32 6-bit float (f6x32_t) values to be converted.
 * @return  The corresponding float representation.
 */
template <>
inline __host__ __device__ float32_t type_convert<float32_t, f6x32_t>(f6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float32_t>(x, 1.0f);
#else
    return slowcast_from_f6<float32_t>(x);
#endif
}

template <>
inline __host__ __device__ float16_t type_convert<float16_t, f6x16_t>(f6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float16_t>(x, 1.0f);
#else
    return slowcast_from_f6<float16_t>(x);
#endif
}

template <>
inline __host__ __device__ float16_t type_convert<float16_t, f6x16_pk_t>(f6x16_pk_t x)
{
    return type_convert<float16_t>(static_cast<f6x16_t>(x));
}

/* float16(half_t) -> fp6
 * single value, vector 16, vector 32 conversion
 * Uses stochastic rounding if CK_USE_SR_F6_CONVERSION is defined,
 * otherwise uses round-to-nearest-even.*/
template <>
inline __host__ __device__ f6_t type_convert<f6_t, half_t>(half_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ f6x32_t type_convert<f6x32_t, half32_t>(half32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ f6x32_pk_t type_convert<f6x32_pk_t, half32_t>(half32_t x)
{
    return static_cast<f6x32_pk_t>(type_convert<f6x32_t>(x));
}

template <>
inline __host__ __device__ f6x16_t type_convert<f6x16_t, half16_t>(half16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ f6x16_pk_t type_convert<f6x16_pk_t, half16_t>(half16_t x)
{
    return static_cast<f6x16_pk_t>(type_convert<f6x16_t>(x));
}

/* float16(half_t) <- fp6
 * single value, vector 16, vector 32 conversion*/
template <>
inline __host__ __device__ half_t type_convert<half_t, f6_t>(f6_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half_t>(x, 1.0f);
#else
    return slowcast_from_f6<half_t>(x);
#endif
}

template <>
inline __host__ __device__ half32_t type_convert<half32_t, f6x32_t>(f6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half32_t>(x, 1.0f);
#else
    return slowcast_from_f6<half32_t>(x);
#endif
}

template <>
inline __host__ __device__ half16_t type_convert<half16_t, f6x16_t>(f6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half16_t>(x, 1.0f);
#else
    return slowcast_from_f6<half16_t>(x);
#endif
}

template <>
inline __host__ __device__ half16_t type_convert<half16_t, f6x16_pk_t>(f6x16_pk_t x)
{
    return type_convert<half16_t>(static_cast<f6x16_t>(x));
}

/* float16(bhalf_t) -> fp6
 * single value, vector 16, vector 32 conversion
 * Uses stochastic rounding if CK_USE_SR_F6_CONVERSION is defined,
 * otherwise uses round-to-nearest-even.*/
template <>
inline __host__ __device__ f6_t type_convert<f6_t, bhalf_t>(bhalf_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ f6x32_t type_convert<f6x32_t, bhalf32_t>(bhalf32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ f6x32_pk_t type_convert<f6x32_pk_t, bhalf32_t>(bhalf32_t x)
{
    return static_cast<f6x32_pk_t>(type_convert<f6x32_t>(x));
}

template <>
inline __host__ __device__ f6x16_t type_convert<f6x16_t, bhalf16_t>(bhalf16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ f6x16_pk_t type_convert<f6x16_pk_t, bhalf16_t>(bhalf16_t x)
{
    return static_cast<f6x16_pk_t>(type_convert<f6x16_t>(x));
}

/* float16(bhalf_t) <- fp6
 * single value, vector 16, vector 32 conversion*/
template <>
inline __host__ __device__ bhalf_t type_convert<bhalf_t, f6_t>(f6_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf_t>(x, 1.0f);
#else
    return slowcast_from_f6<bhalf_t>(x);
#endif
}

template <>
inline __host__ __device__ bhalf32_t type_convert<bhalf32_t, f6x32_t>(f6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf32_t>(x, 1.0f);
#else
    return slowcast_from_f6<bhalf32_t>(x);
#endif
}

template <>
inline __host__ __device__ bhalf16_t type_convert<bhalf16_t, f6x16_t>(f6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf16_t>(x, 1.0f);
#else
    return slowcast_from_f6<bhalf16_t>(x);
#endif
}

template <>
inline __host__ __device__ bhalf16_t type_convert<bhalf16_t, f6x16_pk_t>(f6x16_pk_t x)
{
    return type_convert<bhalf16_t>(static_cast<f6x16_t>(x));
}

/**
 * @brief Converts a scalar or vector of float/float16/bfloat16 to packed 6-bit representation.
 *
 * Divides each input element by the provided scale value, then performs conversion
 * with rounding to nearest even to pack each element into 6 bits of precision.
 *
 * @tparam X Input type (scalar or vector of float/float16/bfloat16 with size 1, 16, or 32)
 * @param x     Input scalar or vector value
 * @param scale Scaling factor applied before conversion
 * @return      Converted Y type storing the compressed 6-bit representation (bf6_t, bf6x16_t,
 * bf6x32_t)
 */
template <typename X>
inline __host__ __device__ auto bf6_convert_rne(X x, float scale = 1.0f) ->
    typename utils::bf6_result_type<X>::type
{
    /* derived return type */
    using Y = typename utils::bf6_result_type<X>::type;

    /* template constraints */
    constexpr int N = scalar_type<X>::vector_size;
    using BaseT     = typename scalar_type<X>::type;
    static_assert(N == 1 || N == 16 || N == 32, "Unsupported vector size");
    static_assert(is_same_v<BaseT, float> || is_same_v<BaseT, half_t> || is_same_v<BaseT, bhalf_t>,
                  "Input type must be float, half_t, or bhalf_t");

#if CK_MX_FP6_CVT_FAST_PATH
    return cast_to_f6_scaled<Y, false>(x, scale);
#else
    return slowcast_to_f6_rne<Y>(x, scale);
#endif
}

/**
 * @brief Converts a scalar or vector of float/float16/bfloat16 to packed 6-bit representation.
 *
 * Divides each input element by the provided scale value, then performs conversion
 * stochastic rounding to pack each element into 6 bits of precision.
 *
 * @tparam X Input type (scalar or vector of float/float16/bfloat16 with size 1, 16, or 32)
 * @param x     Input scalar or vector value
 * @param scale Scaling factor applied before conversion
 * @return      Converted Y type storing the compressed 6-bit representation (bf6_t, bf6x16_t,
 * bf6x32_t)
 */
template <typename X>
inline __host__ __device__ auto bf6_convert_sr(X x, float scale = 1.0f) ->
    typename utils::bf6_result_type<X>::type
{
    /* derived return type */
    using Y = typename utils::bf6_result_type<X>::type;

    /* template constraint */
    constexpr int N = scalar_type<X>::vector_size;
    using BaseT     = typename scalar_type<X>::type;
    static_assert(N == 1 || N == 16 || N == 32, "Unsupported vector size");
    static_assert(is_same_v<BaseT, float> || is_same_v<BaseT, half_t> || is_same_v<BaseT, bhalf_t>,
                  "Input type must be float, half_t, or bhalf_t");

#if CK_MX_FP6_CVT_FAST_PATH
    return cast_to_f6_scaled<Y, true>(x, scale);
#else
    return slowcast_to_f6_sr<Y>(x, scale);
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

template <>
inline __host__ __device__ bf6x32_pk_t type_convert<bf6x32_pk_t, float32_t>(float32_t x)
{
    return static_cast<bf6x32_pk_t>(type_convert<bf6x32_t>(x));
}

template <>
inline __host__ __device__ bf6x16_t type_convert<bf6x16_t, float16_t>(float16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ bf6x16_pk_t type_convert<bf6x16_pk_t, float16_t>(float16_t x)
{
    return static_cast<bf6x16_pk_t>(type_convert<bf6x16_t>(x));
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
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float>(x, 1.0f);
#else
    return slowcast_from_f6<float>(x);
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
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float32_t>(x, 1.0f);
#else
    return slowcast_from_f6<float32_t>(x);
#endif
}

template <>
inline __host__ __device__ float16_t type_convert<float16_t, bf6x16_t>(bf6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float16_t>(x, 1.0f);
#else
    return slowcast_from_f6<float16_t>(x);
#endif
}

template <>
inline __host__ __device__ float16_t type_convert<float16_t, bf6x16_pk_t>(bf6x16_pk_t x)
{
    return type_convert<float16_t>(static_cast<bf6x16_t>(x));
}

/* float16(half_t) -> bf6
 * single value, vector 16, vector 32 conversion
 * Uses stochastic rounding if CK_USE_SR_F6_CONVERSION is defined,
 * otherwise uses round-to-nearest-even.*/
template <>
inline __host__ __device__ bf6_t type_convert<bf6_t, half_t>(half_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ bf6x32_t type_convert<bf6x32_t, half32_t>(half32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ bf6x32_pk_t type_convert<bf6x32_pk_t, half32_t>(half32_t x)
{
    return static_cast<bf6x32_pk_t>(type_convert<bf6x32_t>(x));
}

template <>
inline __host__ __device__ bf6x16_t type_convert<bf6x16_t, half16_t>(half16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ bf6x16_pk_t type_convert<bf6x16_pk_t, half16_t>(half16_t x)
{
    return static_cast<bf6x16_pk_t>(type_convert<bf6x16_t>(x));
}

/* float16(half_t) <- bf6
 * single value, vector 16, vector 32 conversion */
template <>
inline __host__ __device__ half_t type_convert<half_t, bf6_t>(bf6_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half_t>(x, 1.0f);
#else
    return slowcast_from_f6<half_t>(x);
#endif
}

template <>
inline __host__ __device__ half32_t type_convert<half32_t, bf6x32_t>(bf6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half32_t>(x, 1.0f);
#else
    return slowcast_from_f6<half32_t>(x);
#endif
}

template <>
inline __host__ __device__ half16_t type_convert<half16_t, bf6x16_t>(bf6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half16_t>(x, 1.0f);
#else
    return slowcast_from_f6<half16_t>(x);
#endif
}

template <>
inline __host__ __device__ half16_t type_convert<half16_t, bf6x16_pk_t>(bf6x16_pk_t x)
{
    return type_convert<half16_t>(static_cast<bf6x16_t>(x));
}

/* float16(bhalf_t) -> bf6
 * single value, vector 16, vector 32 conversion
 * Uses stochastic rounding if CK_USE_SR_F6_CONVERSION is defined,
 * otherwise uses round-to-nearest-even.*/
template <>
inline __host__ __device__ bf6_t type_convert<bf6_t, bhalf_t>(bhalf_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ bf6x32_t type_convert<bf6x32_t, bhalf32_t>(bhalf32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ bf6x32_pk_t type_convert<bf6x32_pk_t, bhalf32_t>(bhalf32_t x)
{
    return static_cast<bf6x32_pk_t>(type_convert<bf6x32_t>(x));
}

template <>
inline __host__ __device__ bf6x16_t type_convert<bf6x16_t, bhalf16_t>(bhalf16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ bf6x16_pk_t type_convert<bf6x16_pk_t, bhalf16_t>(bhalf16_t x)
{
    return static_cast<bf6x16_pk_t>(type_convert<bf6x16_t>(x));
}

/* float16(bhalf_t) <- bf6
 * single value, vector 16, vector 32 conversion */
template <>
inline __host__ __device__ bhalf_t type_convert<bhalf_t, bf6_t>(bf6_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf_t>(x, 1.0f);
#else
    return slowcast_from_f6<bhalf_t>(x);
#endif
}

template <>
inline __host__ __device__ bhalf32_t type_convert<bhalf32_t, bf6x32_t>(bf6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf32_t>(x, 1.0f);
#else
    return slowcast_from_f6<bhalf32_t>(x);
#endif
}

template <>
inline __host__ __device__ bhalf16_t type_convert<bhalf16_t, bf6x16_t>(bf6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf16_t>(x, 1.0f);
#else
    return slowcast_from_f6<bhalf16_t>(x);
#endif
}

template <>
inline __host__ __device__ bhalf16_t type_convert<bhalf16_t, bf6x16_pk_t>(bf6x16_pk_t x)
{
    return type_convert<bhalf16_t>(static_cast<bf6x16_t>(x));
}
#endif

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
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
#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
