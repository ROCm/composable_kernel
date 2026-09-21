// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type_convert.hpp"
#include "ck/utility/mxf8_utils.hpp"

#ifdef CK_USE_NATIVE_MX_SUPPORT
#define CK_USE_NATIVE_MX_SUPPORT 1
#else
#define CK_USE_NATIVE_MX_SUPPORT 0
#endif

namespace ck {

// Declare a template function for scaled conversion
template <typename Y, typename X>
#if CK_USE_OCP_FP8
__host__ __device__ constexpr Y scaled_type_convert(e8m0_bexp_t scale, X x);
#else
__host__ constexpr Y scaled_type_convert(e8m0_bexp_t scale, X x);
#endif

// convert f8_ocp_t to fp32
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float scaled_type_convert<float, f8_ocp_t>(e8m0_bexp_t scale, f8_ocp_t x)
#else
inline __host__ float scaled_type_convert<float, f8_ocp_t>(e8m0_bexp_t scale, f8_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f32_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.data);
#else
    return type_convert<float>(scale) * type_convert<float>(x);
#endif
}

// convert bf8_ocp_t to fp32
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float scaled_type_convert<float, bf8_ocp_t>(e8m0_bexp_t scale,
                                                                       bf8_ocp_t x)
#else
inline __host__ float scaled_type_convert<float, bf8_ocp_t>(e8m0_bexp_t scale, bf8_ocp_t x)
#endif
{

#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f32_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.data);
#else
    return type_convert<float>(scale) * type_convert<float>(x);
#endif
}

// convert 2 x f8_ocp_t to 2 x fp32
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float2_t scaled_type_convert<float2_t, f8x2_ocp_t>(e8m0_bexp_t scale,
                                                                              f8x2_ocp_t x)
#else
inline __host__ float2_t scaled_type_convert<float2_t, f8x2_ocp_t>(e8m0_bexp_t scale, f8x2_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f32_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x2_storage_t>()[Number<0>{}]);
#else
    auto v_f8x2 = type_convert<float2_t>(x);
    return float2_t{v_f8x2[0] * type_convert<float>(scale), v_f8x2[1] * type_convert<float>(scale)};
#endif
}

// convert 2 x bf8_ocp_t to 2 x fp32
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float2_t scaled_type_convert<float2_t, bf8x2_ocp_t>(e8m0_bexp_t scale,
                                                                               bf8x2_ocp_t x)
#else
inline __host__ float2_t scaled_type_convert<float2_t, bf8x2_ocp_t>(e8m0_bexp_t scale,
                                                                    bf8x2_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f32_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x2_storage_t>()[Number<0>{}]);
#else
    auto v_f8x2 = type_convert<float2_t>(x);
    return float2_t{v_f8x2[0] * type_convert<float>(scale), v_f8x2[1] * type_convert<float>(scale)};
#endif
}

// convert 8 x f8_ocp_t to 8 x fp32
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float8_t scaled_type_convert<float8_t, f8x8_ocp_t>(e8m0_bexp_t scale,
                                                                              f8x8_ocp_t x)
#else
inline __host__ float8_t scaled_type_convert<float8_t, f8x8_ocp_t>(e8m0_bexp_t scale, f8x8_ocp_t x)
#endif
{
    union
    {
        float8_t vf32_8x1;
        float2_t vf32_2x4[4];
        float vf32_1x8[8];
    } out;

#if CK_MX_FP8_CVT_FAST_PATH
    out.vf32_8x1 = fp8_impl::cast_to_f32_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
#else
    union
    {
        f8x8_ocp_t vf8_8x1;
        f8x2_ocp_t vf8_2x4[4];
    } in(x);
    ck::static_for<0, 4, 1>{}(
        [&](auto i) { out.vf32_2x4[i] = scaled_type_convert<float2_t>(scale, in.vf8_2x4[i]); });
#endif
    return out.vf32_8x1;
}

// convert 8 x bf8_ocp_t to 8 x fp32
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float8_t scaled_type_convert<float8_t, bf8x8_ocp_t>(e8m0_bexp_t scale,
                                                                               bf8x8_ocp_t x)
#else
inline __host__ float8_t scaled_type_convert<float8_t, bf8x8_ocp_t>(e8m0_bexp_t scale,
                                                                    bf8x8_ocp_t x)
#endif
{
    union
    {
        float8_t vf32_8x1;
        float2_t vf32_2x4[4];
        float vf32_1x8[8];
    } out;

#if CK_MX_FP8_CVT_FAST_PATH
    out.vf32_8x1 = fp8_impl::cast_to_f32_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
#else
    union
    {
        bf8x8_ocp_t vf8_8x1;
        bf8x2_ocp_t vf8_2x4[4];
    } in(x);
    ck::static_for<0, 4, 1>{}(
        [&](auto i) { out.vf32_2x4[i] = scaled_type_convert<float2_t>(scale, in.vf8_2x4[i]); });
#endif
    return out.vf32_8x1;
}

// convert 16 x f8_ocp_t to 16 x fp32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float16_t scaled_type_convert<float16_t, f8x16_ocp_t>(e8m0_bexp_t scale,
                                                                                 f8x16_ocp_t x)
#else
inline __host__ float16_t scaled_type_convert<float16_t, f8x16_ocp_t>(e8m0_bexp_t scale,
                                                                      f8x16_ocp_t x)
#endif
{
    union
    {
        f8x16_ocp_t f8_16x1;
        f8x8_ocp_t f8_8x2[2];
    } in{x};
    union
    {
        float16_t float_16x1;
        float8_t float_8x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}([&](auto i) {
        out.float_8x2[i] = scaled_type_convert<float8_t, f8x8_ocp_t>(scale, in.f8_8x2[i]);
    });

    return out.float_16x1;
}

// convert 16 x bf8_ocp_t to 16 x fp32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float16_t scaled_type_convert<float16_t, bf8x16_ocp_t>(e8m0_bexp_t scale,
                                                                                  bf8x16_ocp_t x)
#else
inline __host__ float16_t scaled_type_convert<float16_t, bf8x16_ocp_t>(e8m0_bexp_t scale,
                                                                       bf8x16_ocp_t x)
#endif
{
    union
    {
        bf8x16_ocp_t bf8_16x1;
        bf8x8_ocp_t bf8_8x2[8];
    } in{x};
    union
    {
        float16_t float_16x1;
        float8_t float_8x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}([&](auto i) {
        out.float_8x2[i] = scaled_type_convert<float8_t, bf8x8_ocp_t>(scale, in.bf8_8x2[i]);
    });

    return out.float_16x1;
}

// convert 32 x f8_ocp_t to 32 x fp32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float32_t scaled_type_convert<float32_t, f8x32_ocp_t>(e8m0_bexp_t scale,
                                                                                 f8x32_ocp_t x)
#else
inline __host__ float32_t scaled_type_convert<float32_t, f8x32_ocp_t>(e8m0_bexp_t scale,
                                                                      f8x32_ocp_t x)
#endif
{
    union
    {
        f8x32_ocp_t f8_32x1;
        f8x16_ocp_t f8_16x2[2];
    } in{x};
    union
    {
        float32_t float_32x1;
        float16_t float_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}([&](auto i) {
        out.float_16x2[i] = scaled_type_convert<float16_t, f8x16_ocp_t>(scale, in.f8_16x2[i]);
    });

    return out.float_32x1;
}

// convert 32 x bf8_ocp_t to 32 x fp32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float32_t scaled_type_convert<float32_t, bf8x32_ocp_t>(e8m0_bexp_t scale,
                                                                                  bf8x32_ocp_t x)
#else
inline __host__ float32_t scaled_type_convert<float32_t, bf8x32_ocp_t>(e8m0_bexp_t scale,
                                                                       bf8x32_ocp_t x)
#endif
{
    union
    {
        bf8x32_ocp_t bf8_32x1;
        bf8x16_ocp_t bf8_16x2[2];
    } in{x};
    union
    {
        float32_t float_32x1;
        float16_t float_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}([&](auto i) {
        out.float_16x2[i] = scaled_type_convert<float16_t, bf8x16_ocp_t>(scale, in.bf8_16x2[i]);
    });

    return out.float_32x1;
}

// convert fp32 to fp8
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ f8_ocp_t scaled_type_convert<f8_ocp_t, float>(e8m0_bexp_t scale, float x)
#else
inline __host__ f8_ocp_t scaled_type_convert<f8_ocp_t, float>(e8m0_bexp_t scale, float x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32 to bf8
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bf8_ocp_t scaled_type_convert<bf8_ocp_t, float>(e8m0_bexp_t scale,
                                                                           float x)
#else
inline __host__ bf8_ocp_t scaled_type_convert<bf8_ocp_t, float>(e8m0_bexp_t scale, float x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x2 to fp8x2
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ f8x2_ocp_t scaled_type_convert<f8x2_ocp_t, float2_t>(e8m0_bexp_t scale,
                                                                                float2_t x)
#else
inline __host__ f8x2_ocp_t scaled_type_convert<f8x2_ocp_t, float2_t>(e8m0_bexp_t scale, float2_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x2_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x2_ocp_t>(x, type_convert<float>(scale));
#endif
}
// convert fp32x2 to bf8x2
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bf8x2_ocp_t scaled_type_convert<bf8x2_ocp_t, float2_t>(e8m0_bexp_t scale,
                                                                                  float2_t x)
#else
inline __host__ bf8x2_ocp_t scaled_type_convert<bf8x2_ocp_t, float2_t>(e8m0_bexp_t scale,
                                                                       float2_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x2_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x2_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x8 to fp8x8
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ f8x8_ocp_t scaled_type_convert<f8x8_ocp_t, float8_t>(e8m0_bexp_t scale,
                                                                                float8_t x)
#else
inline __host__ f8x8_ocp_t scaled_type_convert<f8x8_ocp_t, float8_t>(e8m0_bexp_t scale, float8_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x8_ocp_t>(x, type_convert<float>(scale));
#endif
}
// convert fp32x8 to bf8x8
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bf8x8_ocp_t scaled_type_convert<bf8x8_ocp_t, float8_t>(e8m0_bexp_t scale,
                                                                                  float8_t x)
#else
inline __host__ bf8x8_ocp_t scaled_type_convert<bf8x8_ocp_t, float8_t>(e8m0_bexp_t scale,
                                                                       float8_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x8_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x16 to fp8x16
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ f8x16_ocp_t
scaled_type_convert<f8x16_ocp_t, float16_t>(e8m0_bexp_t scale, float16_t x)
#else
inline __host__ f8x16_ocp_t scaled_type_convert<f8x16_ocp_t, float16_t>(e8m0_bexp_t scale,
                                                                        float16_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x16_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x16_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x16 to bf8x16
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bf8x16_ocp_t
scaled_type_convert<bf8x16_ocp_t, float16_t>(e8m0_bexp_t scale, float16_t x)
#else
inline __host__ bf8x16_ocp_t scaled_type_convert<bf8x16_ocp_t, float16_t>(e8m0_bexp_t scale,
                                                                          float16_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x16_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x16_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x32 to fp8x32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ f8x32_ocp_t
scaled_type_convert<f8x32_ocp_t, float32_t>(e8m0_bexp_t scale, float32_t x)
#else
inline __host__ f8x32_ocp_t scaled_type_convert<f8x32_ocp_t, float32_t>(e8m0_bexp_t scale,
                                                                        float32_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x32_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x32_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x32 to bf8x32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bf8x32_ocp_t
scaled_type_convert<bf8x32_ocp_t, float32_t>(e8m0_bexp_t scale, float32_t x)
#else
inline __host__ bf8x32_ocp_t scaled_type_convert<bf8x32_ocp_t, float32_t>(e8m0_bexp_t scale,
                                                                          float32_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x32_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x32_ocp_t>(x, type_convert<float>(scale));
#endif
}

// activate for architectures with native MX support
#if CK_USE_NATIVE_MX_SUPPORT
// convert fp4 to fp32
template <>
inline __host__ __device__ float scaled_type_convert<float, f4_t>(e8m0_bexp_t scale, f4_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<float>(x, type_convert<float>(scale));
#else
    return utils::to_float<f4_t>(scale, x);
#endif
}

// convert vector of 2 fp4 to vector of 2 fp32
template <>
inline __host__ __device__ float2_t scaled_type_convert<float2_t, f4x2_t>(e8m0_bexp_t scale,
                                                                          f4x2_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<float2_t>(x, type_convert<float>(scale));
#else
    float2_t ret{utils::to_float<f4_t>(
                     scale, x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{})),
                 utils::to_float<f4_t>(
                     scale, x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}))};
    return ret;
#endif
}

// convert vector of 8 fp4 to vector of 8 fp32
template <>
inline __host__ __device__ float8_t scaled_type_convert<float8_t, f4x8_t>(e8m0_bexp_t scale,
                                                                          f4x8_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<float8_t>(x, type_convert<float>(scale));
#else
    union
    {
        float8_t vf32_8x1;
        float vf32[8];
    } ret{};

    ck::static_for<0, 4, 1>{}([&](auto i) {
        ret.vf32[2 * i] = utils::to_float<f4_t>(
            scale, x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<0>{}));
        ret.vf32[2 * i + 1] = utils::to_float<f4_t>(
            scale, x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<1>{}));
    });
    return ret.vf32_8x1;
#endif
}

// convert vector of 32 fp4 to vector of 32 fp32
template <>
inline __host__ __device__ float32_t scaled_type_convert<float32_t, f4x32_t>(e8m0_bexp_t scale,
                                                                             f4x32_t x)
{
    constexpr int N = 32 / 8;
    union
    {
        f4x32_t f4x32_array;
        f4x8_t v8fp4x4[N];
    } value{x};

    union
    {
        float32_t vf32;
        float8_t v8f32x4[N];
    } ret;

    ck::static_for<0, N, 1>{}([&](auto idx) {
        ret.v8f32x4[idx] = scaled_type_convert<float8_t>(scale, value.v8fp4x4[idx]);
    });
    return ret.vf32;
}

// convert fp32 to fp4
template <>
inline __host__ __device__ f4_t scaled_type_convert<f4_t, float>(e8m0_bexp_t scale, float x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

// convert vector of 2 fp32 to vector of 2 fp4
template <>
inline __host__ __device__ f4x2_t scaled_type_convert<f4x2_t, float2_t>(e8m0_bexp_t scale,
                                                                        float2_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

// convert vector of 8 fp32 to vector of 8 fp4
template <>
inline __host__ __device__ f4x8_t scaled_type_convert<f4x8_t, float8_t>(e8m0_bexp_t scale,
                                                                        float8_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

// convert vector of 32 fp32 to vector of 32 fp4
template <>
inline __host__ __device__ f4x32_t scaled_type_convert<f4x32_t, float32_t>(e8m0_bexp_t scale,
                                                                           float32_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

// float16 <-> fp4
template <>
inline __host__ __device__ half_t scaled_type_convert<half_t, f4_t>(e8m0_bexp_t scale, f4_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<half_t>(x, type_convert<float>(scale));
#else
    return type_convert<half_t>(utils::to_float<f4_t>(scale, x));
#endif
}

template <>
inline __host__ __device__ half2_t scaled_type_convert<half2_t, f4x2_t>(e8m0_bexp_t scale, f4x2_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<half2_t>(x, type_convert<float>(scale));
#else
    return half2_t{type_convert<half_t>(utils::to_float<f4_t>(
                       scale, x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}))),
                   type_convert<half_t>(utils::to_float<f4_t>(
                       scale, x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{})))};
#endif
}

template <>
inline __host__ __device__ half8_t scaled_type_convert<half8_t, f4x8_t>(e8m0_bexp_t scale, f4x8_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<half8_t>(x, type_convert<float>(scale));
#else
    union
    {
        half8_t vf16_8x1;
        half_t vf16[8];
    } ret{};

    ck::static_for<0, 4, 1>{}([&](auto i) {
        ret.vf16[2 * i]     = type_convert<half_t>(utils::to_float<f4_t>(
            scale, x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<0>{})));
        ret.vf16[2 * i + 1] = type_convert<half_t>(utils::to_float<f4_t>(
            scale, x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<1>{})));
    });
    return ret.vf16_8x1;
#endif
}

template <>
inline __host__ __device__ half32_t scaled_type_convert<half32_t, f4x32_t>(e8m0_bexp_t scale,
                                                                           f4x32_t x)
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
        [&](auto idx) { ret.v8f16[idx] = scaled_type_convert<half8_t>(scale, value.v8f4[idx]); });
    return ret.vf16;
}

// convert fp16 to fp4
template <>
inline __host__ __device__ f4_t scaled_type_convert<f4_t, half_t>(e8m0_bexp_t scale, half_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ f4x2_t scaled_type_convert<f4x2_t, half2_t>(e8m0_bexp_t scale, half2_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ f4x8_t scaled_type_convert<f4x8_t, half8_t>(e8m0_bexp_t scale, half8_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ f4x32_t scaled_type_convert<f4x32_t, half32_t>(e8m0_bexp_t scale,
                                                                          half32_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

// bfloat16 <-> fp4
template <>
inline __host__ __device__ bhalf_t scaled_type_convert<bhalf_t, f4_t>(e8m0_bexp_t scale, f4_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<bhalf_t>(x, type_convert<float>(scale));
#else
    return type_convert<bhalf_t>(utils::to_float<f4_t>(scale, x));
#endif
}

template <>
inline __host__ __device__ bhalf2_t scaled_type_convert<bhalf2_t, f4x2_t>(e8m0_bexp_t scale,
                                                                          f4x2_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<bhalf2_t>(x, type_convert<float>(scale));
#else
    return bhalf2_t{type_convert<bhalf_t>(utils::to_float<f4_t>(
                        scale, x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}))),
                    type_convert<bhalf_t>(utils::to_float<f4_t>(
                        scale, x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{})))};
#endif
}

template <>
inline __host__ __device__ bhalf8_t scaled_type_convert<bhalf8_t, f4x8_t>(e8m0_bexp_t scale,
                                                                          f4x8_t x)
{
#if CK_MX_FP4_CVT_FAST_PATH
    return cast_from_f4_scaled<bhalf8_t>(x, type_convert<float>(scale));
#else
    union
    {
        bhalf8_t vf16_8x1;
        bhalf_t vf16[8];
    } ret{};

    ck::static_for<0, 4, 1>{}([&](auto i) {
        ret.vf16[2 * i]     = type_convert<bhalf_t>(utils::to_float<f4_t>(
            scale, x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<0>{})));
        ret.vf16[2 * i + 1] = type_convert<bhalf_t>(utils::to_float<f4_t>(
            scale, x.AsType<f4x2_pk_t>()[Number<i>{}].template unpack<>(Number<1>{})));
    });
    return ret.vf16_8x1;
#endif
}

template <>
inline __host__ __device__ bhalf32_t scaled_type_convert<bhalf32_t, f4x32_t>(e8m0_bexp_t scale,
                                                                             f4x32_t x)
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
        [&](auto idx) { ret.v8f16[idx] = scaled_type_convert<bhalf8_t>(scale, value.v8f4[idx]); });
    return ret.vf16;
}

// convert fp16 to fp4
template <>
inline __host__ __device__ f4_t scaled_type_convert<f4_t, bhalf_t>(e8m0_bexp_t scale, bhalf_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ f4x2_t scaled_type_convert<f4x2_t, bhalf2_t>(e8m0_bexp_t scale,
                                                                        bhalf2_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ f4x8_t scaled_type_convert<f4x8_t, bhalf8_t>(e8m0_bexp_t scale,
                                                                        bhalf8_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ f4x32_t scaled_type_convert<f4x32_t, bhalf32_t>(e8m0_bexp_t scale,
                                                                           bhalf32_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a 6-bit floating-point value (f6_t) to a 32-bit float,
 *        applying the specified scaling factor.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param x     The f6_t value to be converted.
 * @return      The converted 32-bit float representation of the input.
 */
template <>
inline __host__ __device__ float scaled_type_convert<float, f6_t>(e8m0_bexp_t scale, f6_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<float>(x, scale);
#endif
}

/**
 * @brief Converts a vector of 16 6-bit floating-point values (f6x16_t) to a vector of 16 floats,
 *        applying the specified scaling factor.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The f6x16_t vector to be converted.
 * @return      The converted float vector representation of the input.
 */
template <>
inline __host__ __device__ float16_t scaled_type_convert<float16_t, f6x16_t>(e8m0_bexp_t scale,
                                                                             f6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float16_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<float16_t>(x, scale);
#endif
}

/**
 * @brief Converts a vector of 32 6-bit floating-point values (f6x32_t) to a vector of 32 floats,
 *        applying the specified scaling factor.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The f6x32_t vector to be converted.
 * @return      The converted float vector representation of the input.
 */
template <>
inline __host__ __device__ float32_t scaled_type_convert<float32_t, f6x32_t>(e8m0_bexp_t scale,
                                                                             f6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float32_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<float32_t>(x, scale);
#endif
}

/**
 * @brief Converts a 6-bit floating-point value (bf6_t) to a 32-bit float,
 *        applying the specified scaling factor.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param x     The bf6_t value to be converted.
 * @return      The converted 32-bit float representation of the input.
 */
template <>
inline __host__ __device__ float scaled_type_convert<float, bf6_t>(e8m0_bexp_t scale, bf6_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<float>(x, scale);
#endif
}

/**
 * @brief Converts a vector of 6-bit floating-point values (bf6x16_t) to a vector of 16 floats,
 *        applying the specified scaling factor.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The bf6x16_t vector to be converted.
 * @return      The converted vector of 16 float representation of the input.
 */
template <>
inline __host__ __device__ float16_t scaled_type_convert<float16_t, bf6x16_t>(e8m0_bexp_t scale,
                                                                              bf6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float16_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<float16_t>(x, scale);
#endif
}

/**
 * @brief Converts a vector of 6-bit floating-point values (bf6x32_t) to a vector of 32 floats,
 *        applying the specified scaling factor.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The bf6x32_t vector to be converted.
 * @return      The converted vector of 32 float representation of the input.
 */
template <>
inline __host__ __device__ float32_t scaled_type_convert<float32_t, bf6x32_t>(e8m0_bexp_t scale,
                                                                              bf6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<float32_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<float32_t>(x, scale);
#endif
}

/**
 * @brief Converts a 32-bit float to a 6-bit floating-point value (f6_t), applying the specified
 * scale.
 *
 * Depending on whether CK_USE_SR_F6_CONVERSION is defined, it uses either stochastic rounding
 * (f6_convert_sr) or round-to-nearest-even (f6_convert_rne).
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param x     The float value to convert.
 * @return      The converted 6-bit floating-point value (f6_t).
 */
template <>
inline __host__ __device__ f6_t scaled_type_convert<f6_t, float>(e8m0_bexp_t scale, float x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 16 floats to a vector of 16 6-bit floating-point values (f6x16_t),
 * applying the specified scale.
 *
 * Depending on whether CK_USE_SR_F6_CONVERSION is defined, it uses either stochastic rounding
 * (f6_convert_sr) or round-to-nearest-even (f6_convert_rne).
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The float vector to convert.
 * @return      The converted vector of 6-bit floating-point values (f6x16_t).
 */
template <>
inline __host__ __device__ f6x16_t scaled_type_convert<f6x16_t, float16_t>(e8m0_bexp_t scale,
                                                                           float16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 32 floats to a vector of 32 6-bit floating-point values (f6x32_t),
 * applying the specified scale.
 *
 * Depending on whether CK_USE_SR_F6_CONVERSION is defined, it uses either stochastic rounding
 * (f6_convert_sr) or round-to-nearest-even (f6_convert_rne).
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The float vector to convert.
 * @return      The converted vector of 6-bit floating-point values (f6x32_t).
 */
template <>
inline __host__ __device__ f6x32_t scaled_type_convert<f6x32_t, float32_t>(e8m0_bexp_t scale,
                                                                           float32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a 32-bit float to a 6-bit floating-point value (bf6_t), applying the specified
 * scale.
 *
 * Depending on whether CK_USE_SR_F6_CONVERSION is defined, it uses either stochastic rounding
 * (bf6_convert_sr) or round-to-nearest-even (bf6_convert_rne).
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param x     The float value to convert.
 * @return      The converted 6-bit floating-point value (bf6_t).
 */
template <>
inline __host__ __device__ bf6_t scaled_type_convert<bf6_t, float>(e8m0_bexp_t scale, float x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 16 floats to a vector of 16 6-bit floating-point values (bf6x16_t),
 * applying the specified scale.
 *
 * Depending on whether CK_USE_SR_F6_CONVERSION is defined, it uses either stochastic rounding
 * (bf6_convert_sr) or round-to-nearest-even (bf6_convert_rne).
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The float vector to convert.
 * @return      The converted 6-bit floating-point vector (bf6x16_t).
 */
template <>
inline __host__ __device__ bf6x16_t scaled_type_convert<bf6x16_t, float16_t>(e8m0_bexp_t scale,
                                                                             float16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 32 floats to a vector of 32 6-bit floating-point values (bf6x32_t),
 * applying the specified scale.
 *
 * Depending on whether CK_USE_SR_F6_CONVERSION is defined, it uses either stochastic rounding
 * (bf6_convert_sr) or round-to-nearest-even (bf6_convert_rne).
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The float vector to convert.
 * @return      The converted 6-bit floating-point vector (bf6x32_t).
 */
template <>
inline __host__ __device__ bf6x32_t scaled_type_convert<bf6x32_t, float32_t>(e8m0_bexp_t scale,
                                                                             float32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}

/* float16(half_t) <-> bf6
 * single value, vector 16, vector 32 conversion.*/
template <>
inline __host__ __device__ half_t scaled_type_convert<half_t, f6_t>(e8m0_bexp_t scale, f6_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<half_t>(x, scale);
#endif
}

template <>
inline __host__ __device__ half16_t scaled_type_convert<half16_t, f6x16_t>(e8m0_bexp_t scale,
                                                                           f6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half16_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<half16_t>(x, scale);
#endif
}

template <>
inline __host__ __device__ half32_t scaled_type_convert<half32_t, f6x32_t>(e8m0_bexp_t scale,
                                                                           f6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half32_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<half32_t>(x, scale);
#endif
}

template <>
inline __host__ __device__ half_t scaled_type_convert<half_t, bf6_t>(e8m0_bexp_t scale, bf6_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<half_t>(x, scale);
#endif
}

template <>
inline __host__ __device__ half16_t scaled_type_convert<half16_t, bf6x16_t>(e8m0_bexp_t scale,
                                                                            bf6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half16_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<half16_t>(x, scale);
#endif
}

template <>
inline __host__ __device__ half32_t scaled_type_convert<half32_t, bf6x32_t>(e8m0_bexp_t scale,
                                                                            bf6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<half32_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<half32_t>(x, scale);
#endif
}

/* float16(half_t) -> fp6, bf6
 * single value, vector 16, vector 32 conversion
 * Uses stochastic rounding if CK_USE_SR_F6_CONVERSION is defined,
 * otherwise uses round-to-nearest-even.*/
template <>
inline __host__ __device__ f6_t scaled_type_convert<f6_t, half_t>(e8m0_bexp_t scale, half_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ f6x16_t scaled_type_convert<f6x16_t, half16_t>(e8m0_bexp_t scale,
                                                                          half16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ f6x32_t scaled_type_convert<f6x32_t, half32_t>(e8m0_bexp_t scale,
                                                                          half32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ bf6_t scaled_type_convert<bf6_t, half_t>(e8m0_bexp_t scale, half_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ bf6x16_t scaled_type_convert<bf6x16_t, half16_t>(e8m0_bexp_t scale,
                                                                            half16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ bf6x32_t scaled_type_convert<bf6x32_t, half32_t>(e8m0_bexp_t scale,
                                                                            half32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}

/* bfloat16(bhalf_t) <-> bf6
 * single value, vector 16, vector 32 conversion.*/
template <>
inline __host__ __device__ bhalf_t scaled_type_convert<bhalf_t, f6_t>(e8m0_bexp_t scale, f6_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<bhalf_t>(x, scale);
#endif
}

template <>
inline __host__ __device__ bhalf16_t scaled_type_convert<bhalf16_t, f6x16_t>(e8m0_bexp_t scale,
                                                                             f6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf16_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<bhalf16_t>(x, scale);
#endif
}

template <>
inline __host__ __device__ bhalf32_t scaled_type_convert<bhalf32_t, f6x32_t>(e8m0_bexp_t scale,
                                                                             f6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf32_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<bhalf32_t>(x, scale);
#endif
}

template <>
inline __host__ __device__ bhalf_t scaled_type_convert<bhalf_t, bf6_t>(e8m0_bexp_t scale, bf6_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<bhalf_t>(x, scale);
#endif
}

template <>
inline __host__ __device__ bhalf16_t scaled_type_convert<bhalf16_t, bf6x16_t>(e8m0_bexp_t scale,
                                                                              bf6x16_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf16_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<bhalf16_t>(x, scale);
#endif
}

template <>
inline __host__ __device__ bhalf32_t scaled_type_convert<bhalf32_t, bf6x32_t>(e8m0_bexp_t scale,
                                                                              bf6x32_t x)
{
#if CK_MX_FP6_CVT_FAST_PATH
    return cast_from_f6_scaled<bhalf32_t>(x, type_convert<float>(scale));
#else
    return slowcast_from_f6<bhalf32_t>(x, scale);
#endif
}

/* float16(bhalf_t) -> fp6, bf6
 * single value, vector 16, vector 32 conversion
 * Uses stochastic rounding if CK_USE_SR_F6_CONVERSION is defined,
 * otherwise uses round-to-nearest-even.*/
template <>
inline __host__ __device__ f6_t scaled_type_convert<f6_t, bhalf_t>(e8m0_bexp_t scale, bhalf_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ f6x16_t scaled_type_convert<f6x16_t, bhalf16_t>(e8m0_bexp_t scale,
                                                                           bhalf16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ f6x32_t scaled_type_convert<f6x32_t, bhalf32_t>(e8m0_bexp_t scale,
                                                                           bhalf32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ bf6_t scaled_type_convert<bf6_t, bhalf_t>(e8m0_bexp_t scale, bhalf_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ bf6x16_t scaled_type_convert<bf6x16_t, bhalf16_t>(e8m0_bexp_t scale,
                                                                             bhalf16_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}

template <>
inline __host__ __device__ bf6x32_t scaled_type_convert<bf6x32_t, bhalf32_t>(e8m0_bexp_t scale,
                                                                             bhalf32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}
#endif // #if CK_USE_NATIVE_MX_SUPPORT

// Float16
// convert f8_ocp_t to fp16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ half_t scaled_type_convert<half_t, f8_ocp_t>(e8m0_bexp_t scale,
                                                                        f8_ocp_t x)
#else
inline __host__ half_t scaled_type_convert<half_t, f8_ocp_t>(e8m0_bexp_t scale, f8_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f16_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.data);
#else
    return type_convert<half_t>(type_convert<float>(scale) * type_convert<float>(x));
#endif
}

// convert bf8_ocp_t to fp16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ half_t scaled_type_convert<half_t, bf8_ocp_t>(e8m0_bexp_t scale,
                                                                         bf8_ocp_t x)
#else
inline __host__ half_t scaled_type_convert<half_t, bf8_ocp_t>(e8m0_bexp_t scale, bf8_ocp_t x)
#endif
{

#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f16_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.data);
#else
    return type_convert<half_t>(type_convert<float>(scale) * type_convert<float>(x));
#endif
}

// convert 2 x f8_ocp_t to 2 x fp16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ half2_t scaled_type_convert<half2_t, f8x2_ocp_t>(e8m0_bexp_t scale,
                                                                            f8x2_ocp_t x)
#else
inline __host__ half2_t scaled_type_convert<half2_t, f8x2_ocp_t>(e8m0_bexp_t scale, f8x2_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f16_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x2_storage_t>()[Number<0>{}]);
#else
    return half2_t{scaled_type_convert<half_t>(scale, x.AsType<f8_ocp_t>()[Number<0>{}]),
                   scaled_type_convert<half_t>(scale, x.AsType<f8_ocp_t>()[Number<1>{}])};
#endif
}

// convert 2 x bf8_ocp_t to 2 x fp16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ half2_t scaled_type_convert<half2_t, bf8x2_ocp_t>(e8m0_bexp_t scale,
                                                                             bf8x2_ocp_t x)
#else
inline __host__ half2_t scaled_type_convert<half2_t, bf8x2_ocp_t>(e8m0_bexp_t scale, bf8x2_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f16_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x2_storage_t>()[Number<0>{}]);
#else
    return half2_t{scaled_type_convert<half_t>(scale, x.AsType<bf8_ocp_t>()[Number<0>{}]),
                   scaled_type_convert<half_t>(scale, x.AsType<bf8_ocp_t>()[Number<1>{}])};
#endif
}

// convert 8 x f8_ocp_t to 8 x fp16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ half8_t scaled_type_convert<half8_t, f8x8_ocp_t>(e8m0_bexp_t scale,
                                                                            f8x8_ocp_t x)
#else
inline __host__ half8_t scaled_type_convert<half8_t, f8x8_ocp_t>(e8m0_bexp_t scale, f8x8_ocp_t x)
#endif
{
#if CK_MX_ARCH_125
    return fp8_impl::cast_to_f16_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
#else
    union
    {
        half8_t vf16_8x1;
        half2_t vf16_2x4[4];
    } out;

    union
    {
        f8x8_ocp_t vf8_8x1;
        f8x2_ocp_t vf8_2x4[4];
    } in{x};
    ck::static_for<0, 4, 1>{}(
        [&](auto i) { out.vf16_2x4[i] = scaled_type_convert<half2_t>(scale, in.vf8_2x4[i]); });

    return out.vf16_8x1;
#endif
}

// convert 8 x bf8_ocp_t to 8 x fp16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ half8_t scaled_type_convert<half8_t, bf8x8_ocp_t>(e8m0_bexp_t scale,
                                                                             bf8x8_ocp_t x)
#else
inline __host__ half8_t scaled_type_convert<half8_t, bf8x8_ocp_t>(e8m0_bexp_t scale, bf8x8_ocp_t x)
#endif
{
#if CK_MX_ARCH_125
    return fp8_impl::cast_to_f16_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
#else
    union
    {
        half8_t vf16_8x1;
        half2_t vf16_2x4[4];
    } out;

    union
    {
        bf8x8_ocp_t vbf8_8x1;
        bf8x2_ocp_t vbf8_2x4[4];
    } in(x);
    ck::static_for<0, 4, 1>{}(
        [&](auto i) { out.vf16_2x4[i] = scaled_type_convert<half2_t>(scale, in.vbf8_2x4[i]); });

    return out.vf16_8x1;
#endif
}

// convert fp16 to f8_ocp_t
/**
 * @brief Converts a vector of float16s to a vector of 8-bit floating-point values(f8),
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The float16 vector to convert.
 * @return      The converted 8-bit floating-point vector(f8).
 */
template <>
inline __host__ __device__ f8_ocp_t scaled_type_convert<f8_ocp_t, half_t>(e8m0_bexp_t scale,
                                                                          half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8_ocp_t>(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of float16s to a vector of 8-bit floating-point values(bf8),
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The float16 vector to convert.
 * @return      The converted 8-bit floating-point vector(bf8).
 */
template <>
inline __host__ __device__ bf8_ocp_t scaled_type_convert<bf8_ocp_t, half_t>(e8m0_bexp_t scale,
                                                                            half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8_ocp_t>(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 2 float16 to a vector of 2 8-bit floating-point(fp8) values,
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The float16 vector to convert.
 * @return      The converted 8-bit floating-point vector(fp8).
 */
template <>
inline __host__ __device__ f8x2_ocp_t scaled_type_convert<f8x2_ocp_t, half2_t>(e8m0_bexp_t scale,
                                                                               half2_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x2_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x2_ocp_t>(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 2 float16 to a vector of 2 8-bit floating-point(bf8) values,
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The float16 vector to convert.
 * @return      The converted 8-bit floating-point vector(bf8).
 */
template <>
inline __host__ __device__ bf8x2_ocp_t scaled_type_convert<bf8x2_ocp_t, half2_t>(e8m0_bexp_t scale,
                                                                                 half2_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x2_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x2_ocp_t>(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 8 float16 to a vector of 8 8-bit floating-point(fp8) values,
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The float16 vector to convert.
 * @return      The converted 8-bit floating-point vector(fp8).
 */
template <>
inline __host__ __device__ f8x8_ocp_t scaled_type_convert<f8x8_ocp_t, half8_t>(e8m0_bexp_t scale,
                                                                               half8_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x8_ocp_t>(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 8 float16s to a vector of 8 8-bit floating-point values(bf8),
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The float16 vector to convert.
 * @return      The converted 8-bit floating-point vector(bf8).
 */
template <>
inline __host__ __device__ bf8x8_ocp_t scaled_type_convert<bf8x8_ocp_t, half8_t>(e8m0_bexp_t scale,
                                                                                 half8_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x8_ocp_t>(x, type_convert<float>(scale));
#endif
}

// Bfloat16
// convert f8_ocp_t to bf16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bhalf_t scaled_type_convert<bhalf_t, f8_ocp_t>(e8m0_bexp_t scale,
                                                                          f8_ocp_t x)
#else
inline __host__ bhalf_t scaled_type_convert<bhalf_t, f8_ocp_t>(e8m0_bexp_t scale, f8_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_bf16_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.data);
#else
    return type_convert<bhalf_t>(type_convert<float>(scale) * type_convert<float>(x));
#endif
}

// convert bf8_ocp_t to bf16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bhalf_t scaled_type_convert<bhalf_t, bf8_ocp_t>(e8m0_bexp_t scale,
                                                                           bf8_ocp_t x)
#else
inline __host__ bhalf_t scaled_type_convert<bhalf_t, bf8_ocp_t>(e8m0_bexp_t scale, bf8_ocp_t x)
#endif
{

#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_bf16_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.data);
#else
    return type_convert<bhalf_t>(type_convert<float>(scale) * type_convert<float>(x));
#endif
}

// convert 2 x f8_ocp_t to 2 x bf16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bhalf2_t scaled_type_convert<bhalf2_t, f8x2_ocp_t>(e8m0_bexp_t scale,
                                                                              f8x2_ocp_t x)
#else
inline __host__ bhalf2_t scaled_type_convert<bhalf2_t, f8x2_ocp_t>(e8m0_bexp_t scale, f8x2_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_bf16_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x2_storage_t>()[Number<0>{}]);
#else
    return bhalf2_t{scaled_type_convert<bhalf_t>(scale, x.AsType<f8_ocp_t>()[Number<0>{}]),
                    scaled_type_convert<bhalf_t>(scale, x.AsType<f8_ocp_t>()[Number<1>{}])};
#endif
}

// convert 2 x bf8_ocp_t to 2 x bf16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bhalf2_t scaled_type_convert<bhalf2_t, bf8x2_ocp_t>(e8m0_bexp_t scale,
                                                                               bf8x2_ocp_t x)
#else
inline __host__ bhalf2_t scaled_type_convert<bhalf2_t, bf8x2_ocp_t>(e8m0_bexp_t scale,
                                                                    bf8x2_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_bf16_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x2_storage_t>()[Number<0>{}]);
#else
    return bhalf2_t{scaled_type_convert<bhalf_t>(scale, x.AsType<bf8_ocp_t>()[Number<0>{}]),
                    scaled_type_convert<bhalf_t>(scale, x.AsType<bf8_ocp_t>()[Number<1>{}])};
#endif
}

// convert 8 x f8_ocp_t to 8 x bf16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bhalf8_t scaled_type_convert<bhalf8_t, f8x8_ocp_t>(e8m0_bexp_t scale,
                                                                              f8x8_ocp_t x)
#else
inline __host__ bhalf8_t scaled_type_convert<bhalf8_t, f8x8_ocp_t>(e8m0_bexp_t scale, f8x8_ocp_t x)
#endif
{
#if CK_MX_ARCH_125
    return fp8_impl::cast_to_bf16_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
#else
    union
    {
        bhalf8_t vbf16_8x1;
        bhalf2_t vbf16_2x4[4];
    } out;

    union
    {
        f8x8_ocp_t vf8_8x1;
        f8x2_ocp_t vf8_2x4[4];
    } in{x};
    ck::static_for<0, 4, 1>{}(
        [&](auto i) { out.vbf16_2x4[i] = scaled_type_convert<bhalf2_t>(scale, in.vf8_2x4[i]); });

    return out.vbf16_8x1;
#endif
}

// convert 8 x bf8_ocp_t to 8 x bf16
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bhalf8_t scaled_type_convert<bhalf8_t, bf8x8_ocp_t>(e8m0_bexp_t scale,
                                                                               bf8x8_ocp_t x)
#else
inline __host__ bhalf8_t scaled_type_convert<bhalf8_t, bf8x8_ocp_t>(e8m0_bexp_t scale,
                                                                    bf8x8_ocp_t x)
#endif
{
#if CK_MX_ARCH_125
    return fp8_impl::cast_to_bf16_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
#else
    union
    {
        bhalf8_t vbf16_8x1;
        bhalf2_t vbf16_2x4[4];
    } out;

    union
    {
        bf8x8_ocp_t vbf8_8x1;
        bf8x2_ocp_t vbf8_2x4[4];
    } in(x);
    ck::static_for<0, 4, 1>{}(
        [&](auto i) { out.vbf16_2x4[i] = scaled_type_convert<bhalf2_t>(scale, in.vbf8_2x4[i]); });

    return out.vbf16_8x1;
#endif
}

// convert bfp16 to f8_ocp_t
/**
 * @brief Converts a vector of bfloat16 to a vector of 8-bit floating-point values(f8),
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The bfloat16 vector to convert.
 * @return      The converted 8-bit floating-point vector(f8).
 */
template <>
inline __host__ __device__ f8_ocp_t scaled_type_convert<f8_ocp_t, bhalf_t>(e8m0_bexp_t scale,
                                                                           bhalf_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8_ocp_t>(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of bfloat16 to a vector of 8-bit floating-point values(bf8),
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The bfloat16 vector to convert.
 * @return      The converted 8-bit floating-point vector(bf8).
 */
template <>
inline __host__ __device__ bf8_ocp_t scaled_type_convert<bf8_ocp_t, bhalf_t>(e8m0_bexp_t scale,
                                                                             bhalf_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8_ocp_t>(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 2 bfloat16 to a vector of 2 8-bit floating-point(fp8) values,
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The bfloat16 vector to convert.
 * @return      The converted 8-bit floating-point vector(fp8).
 */
template <>
inline __host__ __device__ f8x2_ocp_t scaled_type_convert<f8x2_ocp_t, bhalf2_t>(e8m0_bexp_t scale,
                                                                                bhalf2_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x2_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x2_ocp_t>(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 2 bfloat16 to a vector of 2 8-bit floating-point(bf8) values,
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The bfloat16 vector to convert.
 * @return      The converted 8-bit floating-point vector(bf8).
 */
template <>
inline __host__ __device__ bf8x2_ocp_t scaled_type_convert<bf8x2_ocp_t, bhalf2_t>(e8m0_bexp_t scale,
                                                                                  bhalf2_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x2_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x2_ocp_t>(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 8 bfloat16 to a vector of 8 8-bit floating-point(fp8) values,
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The bfloat16 vector to convert.
 * @return      The converted 8-bit floating-point vector(fp8).
 */
template <>
inline __host__ __device__ f8x8_ocp_t scaled_type_convert<f8x8_ocp_t, bhalf8_t>(e8m0_bexp_t scale,
                                                                                bhalf8_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x8_ocp_t>(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 8 bfloat16 to a vector of 8 8-bit floating-point values(bf8),
 * applying the specified scale.
 * *
 * @param scale The exponent scale factor (e8m0_bexp_t ).
 * @param x     The bfloat16 vector to convert.
 * @return      The converted 8-bit floating-point vector(bf8).
 */
template <>
inline __host__ __device__ bf8x8_ocp_t scaled_type_convert<bf8x8_ocp_t, bhalf8_t>(e8m0_bexp_t scale,
                                                                                  bhalf8_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x8_ocp_t>(x, type_convert<float>(scale));
#endif
}

#if CK_MX_ARCH_125
// Declare a template function for wave-wise scaled conversion
/* scale is packed 4 form, see details for FP8/BF8, FP4, FP6 */
template <typename Y, typename X, int Scale_sel>
struct pk4scaled_type_convert_impl
{
    __device__ static constexpr Y run(uint32_t scale, X x);
};

template <typename Y, typename X, int Scale_sel = 0>
__device__ constexpr Y pk4scaled_type_convert(uint32_t scale, X x)
{
    return pk4scaled_type_convert_impl<Y, X, Scale_sel>::run(scale, x);
}

/* scale is packed 4 form [FP8/BF8]
 * Scale_sel: select different scale set and apply to the tensor[16x16] represented by a wave,
 *            th[0-15]: 16x8 and th[16-31]: 16x8
 *      Block 32 :
 *      0(0000): src[th[0:31]]  * scale[th[0:15]][7:0]
 *      1(0001): src[th[0:31]]  * scale[th[16:31]][7:0]
 *      2(0010): src[th[0:31]]  * scale[th[0:15]][23:16]
 *      3(0011): src[th[0:31]]  * scale[th[16:31]][23:16]
 *      4(0100): src[th[0:31]]  * scale[th[0:15]][15:8]
 *      5(0101): src[th[0:31]]  * scale[th[16:31]][15:8]
 *      6(0110): src[th[0:31]]  * scale[th[0:15]][31:24]
 *      7(0111): src[th[0:31]]  * scale[th[16:31]][31:24]
 *      Block 16 : Available for certain revision
 *      8(1000) : src[th[0:15]]  * scale[th[0:15]][7:0]
 *                src[th[16:31]] * scale[th[0:15]][15:8]
 *      9(1001) : src[th[0:15]]  * scale[th[16:31]][7:0]
 *                src[th[16:31]] * scale[th[16:31]][15:8]
 *      10(1010): src[th[0:15]]  * scale[th[0:15]][23:16]
 *                src[th[16:31]] * scale[th[0:15]][31:24]
 *      11(1011): src[th[0:15]]  * scale[th[16:31]][23:16]
 *                src[th[16:31]] * scale[th[16:31]][31:24] */
// float16
template <int Scale_sel>
struct pk4scaled_type_convert_impl<half8_t, f8x8_ocp_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 8 8-bit floating-point(fp8) values to a vector of 8 float16,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(fp8) to convert.
     * @return      The converted float16 vector.
     */
    __device__ static half8_t run(uint32_t scale, f8x8_ocp_t x)
    {
        return fp8_impl::
            cast_to_f16_from_f8_scaled<f8_ocp_t::default_interpret, uint32_t, Scale_sel>(
                scale, x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
    }
};

template <int Scale_sel>
struct pk4scaled_type_convert_impl<half8_t, bf8x8_ocp_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 8 8-bit floating-point(bf8) values to a vector of 8 float16,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(bf8) to convert.
     * @return      The converted float16 vector.
     */
    __device__ static half8_t run(uint32_t scale, bf8x8_ocp_t x)
    {
        return fp8_impl::
            cast_to_f16_from_f8_scaled<bf8_ocp_t::default_interpret, uint32_t, Scale_sel>(
                scale, x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
    }
};

// bfloat16
template <int Scale_sel>
struct pk4scaled_type_convert_impl<bhalf8_t, f8x8_ocp_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 8 8-bit floating-point(fp8) values to a vector of 8 bfloat16,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(fp8) to convert.
     * @return      The converted bfloat16 vector.
     */
    __device__ static bhalf8_t run(uint32_t scale, f8x8_ocp_t x)
    {
        return fp8_impl::
            cast_to_bf16_from_f8_scaled<f8_ocp_t::default_interpret, uint32_t, Scale_sel>(
                scale, x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
    }
};

template <int Scale_sel>
struct pk4scaled_type_convert_impl<bhalf8_t, bf8x8_ocp_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 8 8-bit floating-point(bf8) values to a vector of 8 bfloat16,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(bf8) to convert.
     * @return      The converted bfloat16 vector.
     */
    __device__ static bhalf8_t run(uint32_t scale, bf8x8_ocp_t x)
    {
        return fp8_impl::
            cast_to_bf16_from_f8_scaled<bf8_ocp_t::default_interpret, uint32_t, Scale_sel>(
                scale, x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
    }
};

// float32
template <int Scale_sel>
struct pk4scaled_type_convert_impl<float8_t, f8x8_ocp_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 8 8-bit floating-point(fp8) values to a vector of 8 float32,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(fp8) to convert.
     * @return      The converted float32 vector.
     */
    __device__ static float8_t run(uint32_t scale, f8x8_ocp_t x)
    {
        return fp8_impl::
            cast_to_f32_from_f8_scaled<f8_ocp_t::default_interpret, uint32_t, Scale_sel>(
                scale, x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
    }
};

template <int Scale_sel>
struct pk4scaled_type_convert_impl<float8_t, bf8x8_ocp_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 8 8-bit floating-point(bfp8) values to a vector of 8 float32,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(bfp8) to convert.
     * @return      The converted float32 vector.
     */
    __device__ static float8_t run(uint32_t scale, bf8x8_ocp_t x)
    {
        return fp8_impl::
            cast_to_f32_from_f8_scaled<bf8_ocp_t::default_interpret, uint32_t, Scale_sel>(
                scale, x.AsType<fp8_impl::fp8x8_storage_t>()[Number<0>{}]);
    }
};

/* scale is packed 4 form [FP4]
 * Scale_sel: select different scale set and apply to the tensor[16x16] represented by a wave,
 *            th[0-15]: 16x8 and th[16-31]: 16x8
 *      Block 32 :
 *      0(000): src[th[0-15]]  * scale[th[0-15]][7:0]
                src[th[16-31]] * scale[th[0-15]][15:8]
 *      1(001): src[th[0-15]]  * scale[th[16-31]][7:0]
                src[th[16-31]] * scale[th[16-31]][15:8]
 *      2(010): src[th[0-15]]  * scale[th[0-15]][23:16]
                src[th[16-31]] * scale[th[0-15]][31:24]
 *      3(011): src[th[0-15]]  * scale[th[16-31]][23:16]
                src[th[16-31]] * scale[th[16-31]][31:24]
 *      Block 16 : Available for certain revision
 *      4(100): src[th[0-15]]  * scale[th[0-15]][7:0]
                src[th[16-31]] * scale[th[0-15]][23:16]
 *      5(101): src[th[0-15]]  * scale[th[16-31]][7:0]
                src[th[16-31]] * scale[th[16-31]][23:16]
 *      6(110): src[th[0-15]]  * scale[th[0-15]][15:8]
                src[th[16-31]] * scale[th[0-15]][31:24]
 *      7(111): src[th[0-15]]  * scale[th[16-31]][15:8]
                src[th[16-31]] * scale[th[16-31]][31:24]
 */
// FP4 to float
template <int Scale_sel>
struct pk4scaled_type_convert_impl<float8_t, f4x8_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 8 4-bit floating-point(fp4) values to a vector of 8 float32,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(fp4) to convert.
     * @return      The converted float32 vector.
     */
    __device__ static float8_t run(uint32_t scale, f4x8_t x)
    {
        return cast_from_f4_scaled<float8_t, uint32_t, Scale_sel>(x, scale);
    }
};

// FP4 to float16
template <int Scale_sel>
struct pk4scaled_type_convert_impl<half8_t, f4x8_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 8 4-bit floating-point(fp4) values to a vector of 8 float16,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(fp4) to convert.
     * @return      The converted float16 vector.
     */
    __device__ static half8_t run(uint32_t scale, f4x8_t x)
    {
        return cast_from_f4_scaled<half8_t, uint32_t, Scale_sel>(x, scale);
    }
};

// FP4 to bfloat16
template <int Scale_sel>
struct pk4scaled_type_convert_impl<bhalf8_t, f4x8_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 8 4-bit floating-point(fp4) values to a vector of 8 bfloat16,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(fp4) to convert.
     * @return      The converted bfloat16 vector.
     */
    __device__ static bhalf8_t run(uint32_t scale, f4x8_t x)
    {
        return cast_from_f4_scaled<bhalf8_t, uint32_t, Scale_sel>(x, scale);
    }
};

/* scale is packed 4 form [FP6]
 * Scale_sel: select different scale set and apply to the tensor[16x32] represented by a wave,
 *            th[0-15]: 16x16 and th[16-31]: 16x16
 *      Block 32 :
 *      0(000): src[th[0-15]]  * scale[th[0-15]][7:0]
                src[th[16-31]] * scale[th[0-15]][15:8]
 *      1(001): src[th[0-15]]  * scale[th[16-31]][7:0]
                src[th[16-31]] * scale[th[16-31]][15:8]
 *      2(010): src[th[0-15]]  * scale[th[0-15]][23:16]
                src[th[16-31]] * scale[th[0-15]][31:24]
 *      3(011): src[th[0-15]]  * scale[th[16-31]][23:16]
                src[th[16-31]] * scale[th[16-31]][31:24]
 *      Block 16 : Available for certain revision
 *      4(100): src[th[0-15]]  * scale[th[0-15]][7:0]
                src[th[16-31]] * scale[th[0-15]][23:16]
 *      5(101): src[th[0-15]]  * scale[th[16-31]][7:0]
                src[th[16-31]] * scale[th[16-31]][23:16]
 *      6(110): src[th[0-15]]  * scale[th[0-15]][15:8]
                src[th[16-31]] * scale[th[0-15]][31:24]
 *      7(111): src[th[0-15]]  * scale[th[16-31]][15:8]
                src[th[16-31]] * scale[th[16-31]][31:24]
 */
template <int Scale_sel>
struct pk4scaled_type_convert_impl<float16_t, f6x16_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 16 6-bit floating-point(fp6) values to a vector of 16 float32,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(fp6) to convert.
     * @return      The converted float32 vector.
     */
    __device__ static float16_t run(uint32_t scale, f6x16_t x)
    {
        return cast_from_f6_scaled<float16_t, uint32_t, Scale_sel>(x, scale);
    }
};

template <int Scale_sel>
struct pk4scaled_type_convert_impl<float16_t, bf6x16_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 16 6-bit floating-point(bf6) values to a vector of 16 float32,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(bf6) to convert.
     * @return      The converted float32 vector.
     */
    __device__ static float16_t run(uint32_t scale, bf6x16_t x)
    {
        return cast_from_f6_scaled<float16_t, uint32_t, Scale_sel>(x, scale);
    }
};
// half_t
template <int Scale_sel>
struct pk4scaled_type_convert_impl<half16_t, f6x16_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 16 6-bit floating-point(fp6) values to a vector of 16 float16,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(fp6) to convert.
     * @return      The converted float16 vector.
     */
    __device__ static half16_t run(uint32_t scale, f6x16_t x)
    {
        return cast_from_f6_scaled<half16_t, uint32_t, Scale_sel>(x, scale);
    }
};

template <int Scale_sel>
struct pk4scaled_type_convert_impl<half16_t, bf6x16_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 16 6-bit floating-point(bf6) values to a vector of 16 float16,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(bf6) to convert.
     * @return      The converted float16 vector.
     */
    __device__ static half16_t run(uint32_t scale, bf6x16_t x)
    {
        return cast_from_f6_scaled<half16_t, uint32_t, Scale_sel>(x, scale);
    }
};

// bhalf_t
template <int Scale_sel>
struct pk4scaled_type_convert_impl<bhalf16_t, f6x16_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 16 6-bit floating-point(fp6) values to a vector of 16 float16,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(fp6) to convert.
     * @return      The converted float16 vector.
     */
    __device__ static bhalf16_t run(uint32_t scale, f6x16_t x)
    {
        return cast_from_f6_scaled<bhalf16_t, uint32_t, Scale_sel>(x, scale);
    }
};

template <int Scale_sel>
struct pk4scaled_type_convert_impl<bhalf16_t, bf6x16_t, Scale_sel>
{
    /**
     * @brief Converts a vector of 16 6-bit floating-point(bf6) values to a vector of 16 float16,
     * applying a packed-4 scale.
     * *
     * @param scale The packed-4 exponent scale factor (uint32_t).
     * @param x     The floating-point vector(bf6) to convert.
     * @return      The converted float16 vector.
     */
    __device__ static bhalf16_t run(uint32_t scale, bf6x16_t x)
    {
        return cast_from_f6_scaled<bhalf16_t, uint32_t, Scale_sel>(x, scale);
    }
};
#endif // #if CK_MX_ARCH_125

} // namespace ck
