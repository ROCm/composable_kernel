// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/type_convert.hpp"

namespace ck {

// Declare a template function for scaled conversion
template <typename Y, typename X>
__host__ __device__ constexpr Y scaled_type_convert(e8m0_bexp_t scale, X x);

// convert fp4 to fp32
template <>
inline __host__ __device__ float scaled_type_convert<float, f4_t>(e8m0_bexp_t scale, f4_t x)
{
#if defined(__gfx950__)
    union
    {
        float float_array[2];
        float2_t float2_array;
    } float_values{};
    float_values.float2_array =
        __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, type_convert<float>(scale), 0);
    return float_values.float_array[0];
#else
    return utils::to_float<f4_t>(scale, x);
#endif
}

// convert vector of 2 fp4 to vector of 2 fp32
template <>
inline __host__ __device__ float2_t scaled_type_convert<float2_t, f4x2_t>(e8m0_bexp_t scale,
                                                                          f4x2_t x)
{
#if defined(__gfx950__)
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{};
    value.f4x2_array[0] = x;
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.bitwise, type_convert<float>(scale), 0);
#else
    float2_t ret{utils::to_float<f4_t>(
                     scale, x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{})),
                 utils::to_float<f4_t>(
                     scale, x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}))};
    return ret;
#endif
}

// convert vector of 32 fp4 to vector of 32 fp32
template <>
inline __host__ __device__ float32_t scaled_type_convert<float32_t, f4x32_t>(e8m0_bexp_t scale,
                                                                             f4x32_t x)
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
        scale,
        f4_values.f4x2_array[0].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[1] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[0].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[2] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[1].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[3] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[1].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[4] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[2].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[5] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[2].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[6] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[3].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[7] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[3].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));

    float_values.float_array[0] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[4].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[1] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[4].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[2] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[5].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[3] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[5].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[4] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[6].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[5] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[6].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[6] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[7].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[7] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[7].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));

    float_values.float_array[0] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[8].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[1] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[8].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[2] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[9].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[3] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[9].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[4] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[10].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[5] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[10].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[6] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[11].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[7] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[11].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));

    float_values.float_array[0] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[12].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[1] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[12].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[2] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[13].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[3] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[13].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[4] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[14].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[5] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[14].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));
    float_values.float_array[6] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[15].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{}));
    float_values.float_array[7] = utils::to_float<f4_t>(
        scale,
        f4_values.f4x2_array[15].template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}));

    return float_values.float32_array;
#endif
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
    // currently there is no native conversion instruction
    return utils::to_float<f6_t>(scale, x);
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
    // currently there is no native conversion instruction
    return utils::to_float<bf6_t>(scale, x);
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

} // namespace ck
