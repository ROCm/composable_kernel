// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_CODE_GEN_RTC
#pragma once

#include "ck/utility/numeric_limits.hpp"
#include "ck/utility/mxfp_utils.hpp"
#include "dtype_vector.hpp"

#if CK_MX_ARCH_950 || CK_MX_ARCH_125
#define CK_MX_FP4_CVT_FAST_PATH 1
#else
#define CK_MX_FP4_CVT_FAST_PATH 0
#endif

namespace ck {
namespace utils {

template <>
__host__ __device__ inline bool is_nan<f4_t>(e8m0_bexp_t const scale,
                                             f4_t const dataBytes [[maybe_unused]])
{
    // no need to check for data as it does not have NaN representation
    return scale.is_nan();
}

// no infinity representation in ocp_e2m1_mxfp4 will always return false
template <>
__host__ __device__ inline bool is_inf<f4_t>(e8m0_bexp_t const scale [[maybe_unused]],
                                             f4_t const data [[maybe_unused]])
{
    // no inf representation for ocp_e2m1_mxfp4
    return false;
}

template <>
__host__ __device__ inline bool is_zero<f4_t>(e8m0_bexp_t const scale [[maybe_unused]],
                                              f4_t const data)
{
    // no need to check for scale as it does not have a 0 representation
    f4_t result = (data & 0b00001111) & NumericUtils<f4_t>::set_sign_mask;

    return result == 0b0;
}

template <>
__host__ __device__ inline float to_float<f4_t>(e8m0_bexp_t const scale, f4_t const data)
{
    if(is_nan<f4_t>(scale, data))
        return NumericLimits<float>::QuietNaN();

    if(is_zero<f4_t>(scale, data))
        return (data & NumericUtils<f4_t>::negative_zero_mask) ? -0.0f : 0.0f;

    f4_t prepared_data = data & 0b00001111;

    int scale_exp = get_exponent_value<e8m0_bexp_t>(scale);

    return convert_to_float<f4_t>(prepared_data, scale_exp);
}

template <>
__host__ __device__ inline f4_t sat_convert_to_type<f4_t>(float value)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
    {

        return sign ? NumericUtils<f4_t>::data_max_negative_normal_mask
                    : NumericUtils<f4_t>::data_max_positive_normal_mask;
    }

    if(std::abs(value) > NumericLimits<f4_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<f4_t>::data_max_negative_normal_mask
                    : NumericUtils<f4_t>::data_max_positive_normal_mask;

    f4_t res = convert_to_type<f4_t>(value);

    if(std::abs(to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<f4_t>::DataMinSubnorm())
        return sign ? NumericUtils<f4_t>::negative_zero_mask
                    : NumericUtils<f4_t>::positive_zero_mask;

    return res;
}

template <>
__host__ __device__ inline f4_t sat_convert_to_type_sr<f4_t>(float value, uint32_t seed)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
        return sign ? NumericUtils<f4_t>::data_max_negative_normal_mask
                    : NumericUtils<f4_t>::data_max_positive_normal_mask;

    if(std::abs(value) > NumericLimits<f4_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<f4_t>::data_max_negative_normal_mask
                    : NumericUtils<f4_t>::data_max_positive_normal_mask;

    f4_t res = convert_to_type_sr<f4_t>(value, seed);

    if(std::abs(to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<f4_t>::DataMinSubnorm())
        return sign ? NumericUtils<f4_t>::negative_zero_mask
                    : NumericUtils<f4_t>::positive_zero_mask;

    return res;
}
} // namespace utils

#if CK_MX_FP4_CVT_FAST_PATH
// declare
template <typename T>
static inline __device__ enable_if_t<scalar_type<T>::vector_size == 1, T>
cast_from_f4_scaled(f4_t x, float scale = 1.f);

template <typename T>
static inline __device__ enable_if_t<scalar_type<T>::vector_size == 2, T>
cast_from_f4_scaled(f4x2_t x, float scale = 1.f);

template <typename T, typename Ts = float, int Opsel = 0>
static inline __device__ enable_if_t<scalar_type<T>::vector_size == 8, T>
cast_from_f4_scaled(f4x8_t x, Ts scale = 1.f);

template <typename T,
          bool stochastic_rounding                            = false,
          enable_if_t<scalar_type<T>::vector_size == 1, bool> = true>
static inline __device__ f4_t cast_to_f4_scaled(T x, float scale = 1.f);

template <typename T,
          bool stochastic_rounding                            = false,
          enable_if_t<scalar_type<T>::vector_size == 2, bool> = true>
static inline __device__ f4x2_t cast_to_f4_scaled(T x, float scale = 1.f);

template <typename T,
          bool stochastic_rounding                            = false,
          enable_if_t<scalar_type<T>::vector_size == 8, bool> = true>
static inline __device__ f4x8_t cast_to_f4_scaled(T x, float scale = 1.f);

// definition
#if CK_MX_ARCH_950
// from f4
template <typename T>
static inline __device__ enable_if_t<scalar_type<T>::vector_size == 1, T>
cast_from_f4_scaled(f4_t x, float scale)
{
    using BaseT = typename scalar_type<T>::type;
    using T2    = typename ck::vector_type<BaseT, 2>::type;
    union
    {
        T v_arr[2];
        T2 v2;
    } ret{};
    f4x2_t x2 = x;
    ret.v2    = cast_from_f4_scaled<T2>(x2, scale);

    return ret.v_arr[0];
}

template <typename T>
static inline __device__ enable_if_t<scalar_type<T>::vector_size == 2, T>
cast_from_f4_scaled(f4x2_t x, float scale)
{
    using BaseT = typename scalar_type<T>::type;
    if constexpr(is_same_v<BaseT, float>)
        return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, scale, 0);
    else if constexpr(is_same_v<BaseT, half_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(x, scale, 0);
    else if constexpr(is_same_v<BaseT, bhalf_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(x, scale, 0);
    else
        static_assert(false_type::value, "Unsupported type.");
}

template <typename T, typename Ts, int Opsel>
static inline __device__ enable_if_t<scalar_type<T>::vector_size == 8, T>
cast_from_f4_scaled(f4x8_t x, Ts scale)
{
    static_assert(is_same_v<Ts, float>, "Ts must be float");
    using BaseT         = typename scalar_type<T>::type;
    using T2            = typename ck::vector_type<BaseT, 2>::type;
    constexpr int Npack = scalar_type<T>::vector_size / 2;
    union
    {
        f4x8_t vf4;
        f4x2_t v2f4_arr[Npack];
    } value{x};
    union
    {
        T vec;
        T2 v2_arr[Npack];
    } ret{};

    ck::static_for<0, Npack, 1>{}(
        [&](auto idx) { ret.v2_arr[idx] = cast_from_f4_scaled<T2>(value.v2f4_arr[idx], scale); });
    return ret.vec;
}

// to f4
template <typename T, bool stochastic_rounding, enable_if_t<scalar_type<T>::vector_size == 1, bool>>
static inline __device__ f4_t cast_to_f4_scaled(T x, float scale)
{
    using BaseT = typename scalar_type<T>::type;
    using T2    = typename ck::vector_type<BaseT, 2>::type;
    union
    {
        f4_t f4_array[4];
        f4x2_t f4x2_array[4];
    } value{};

    T2 x2{x, x};
    value.f4x2_array[0] = cast_to_f4_scaled<T2, stochastic_rounding>(x2, scale);
    return value.f4_array[0];
}

template <typename T, bool stochastic_rounding, enable_if_t<scalar_type<T>::vector_size == 2, bool>>
static inline __device__ f4x2_t cast_to_f4_scaled(T x, float scale)
{
    using BaseT = typename scalar_type<T>::type;
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};

    if constexpr(stochastic_rounding)
    {
        uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() *
                                                 (get_thread_global_1d_id() + 1));

        if constexpr(is_same_v<BaseT, float>)
            value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
                value.bitwise, float2_t{x}, rng, scale, 0);
        else if constexpr(is_same_v<BaseT, half_t>)
            value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16(
                value.bitwise, half2_t{x}, rng, scale, 0);
        else if constexpr(is_same_v<BaseT, bhalf_t>)
            value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16(
                value.bitwise, bhalf2_t{x}, rng, scale, 0);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(is_same_v<BaseT, float>)
            value.bitwise =
                __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(value.bitwise, x[0], x[1], scale, 0);
        else if constexpr(is_same_v<BaseT, half_t>)
            value.bitwise =
                __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(value.bitwise, half2_t{x}, scale, 0);
        else if constexpr(is_same_v<BaseT, bhalf_t>)
            value.bitwise =
                __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(value.bitwise, bhalf2_t{x}, scale, 0);
        else
            static_assert(false_type::value, "Unsupported type.");
    }

    return value.f4x2_array[0];
}

template <typename T, bool stochastic_rounding, enable_if_t<scalar_type<T>::vector_size == 8, bool>>
static inline __device__ f4x8_t cast_to_f4_scaled(T x, float scale)
{
    using BaseT         = typename scalar_type<T>::type;
    using T2            = typename ck::vector_type<BaseT, 2>::type;
    constexpr int Npack = scalar_type<T>::vector_size / 2;
    union
    {
        f4x8_t vf4;
        f4x2_t v2f4_arr[Npack];
    } ret{};
    union
    {
        T vec;
        T2 v2_arr[Npack];
    } value{x};

    ck::static_for<0, Npack, 1>{}([&](auto idx) {
        ret.v2f4_arr[idx] = cast_to_f4_scaled<T2, stochastic_rounding>(value.v2_arr[idx], scale);
    });
    return ret.vf4;
}

#elif CK_MX_ARCH_125
// from f4
template <typename T>
static inline __device__ enable_if_t<scalar_type<T>::vector_size == 1, T>
cast_from_f4_scaled(f4_t x, float scale)
{
    using BaseT = typename scalar_type<T>::type;
    using T8    = typename ck::vector_type<BaseT, 8>::type;
    union
    {
        T v_arr[8];
        typename ck::vector_type<BaseT, 8>::type v8;
    } ret{};
    union
    {
        f4_t vf4_arr[4];
        f4x8_t v8f4;
    } value{};
    value.vf4_arr[0] = x;

    ret.v8 = cast_from_f4_scaled<T8>(value.v8f4, scale);
    return ret.v_arr[0];
}

template <typename T>
static inline __device__ enable_if_t<scalar_type<T>::vector_size == 2, T>
cast_from_f4_scaled(f4x2_t x, float scale)
{
    using BaseT = typename scalar_type<T>::type;
    using T8    = typename ck::vector_type<BaseT, 8>::type;
    union
    {
        T v_arr[4];
        typename ck::vector_type<BaseT, 8>::type v8;
    } ret{};

    union
    {
        f4x2_t v2f4_arr[4];
        f4x8_t v8f4;
    } value{};
    value.v2f4_arr[0] = x;

    ret.v8 = cast_from_f4_scaled<T8>(value.v8f4, scale);
    return ret.v_arr[0];
}

template <typename T, typename Ts, int Opsel>
static inline __device__ enable_if_t<scalar_type<T>::vector_size == 8, T>
cast_from_f4_scaled(f4x8_t x, Ts scale)
{
    static_assert(sizeof(Ts) == 4, "Ts must be float or uint32_t");
    using BaseT     = typename scalar_type<T>::type;
    uint32_t scale4 = (ck::is_same_v<Ts, float>)
                          ? bit_cast<uint32_t>(utils::get_exponent_value(e8m0_bexp_t(scale)))
                          : bit_cast<uint32_t>(scale);

    if constexpr(is_same_v<BaseT, float>)
        return __builtin_amdgcn_cvt_scale_pk8_f32_fp4(ck::bit_cast<uint32_t>(x), scale4, Opsel);
    else if constexpr(is_same_v<BaseT, half_t>)
        return __builtin_amdgcn_cvt_scale_pk8_f16_fp4(ck::bit_cast<uint32_t>(x), scale4, Opsel);
    else if constexpr(is_same_v<BaseT, bhalf_t>)
        return __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(ck::bit_cast<uint32_t>(x), scale4, Opsel);
    else
        static_assert(false_type::value, "Unsupported type.");
}

// to f4
template <typename T, bool stochastic_rounding, enable_if_t<scalar_type<T>::vector_size == 1, bool>>
static inline __device__ f4_t cast_to_f4_scaled(T x, float scale)
{
    using BaseT = typename scalar_type<T>::type;
    using T8    = typename ck::vector_type<BaseT, 8>::type;
    union
    {
        f4x8_t v8f4;
        uint8_t i8_array[4];
    } value{0};

    value.v8f4 = cast_to_f4_scaled<T8, stochastic_rounding>(T8(x), scale);
    return value.i8_array[0] & 0b00001111;
}

template <typename T, bool stochastic_rounding, enable_if_t<scalar_type<T>::vector_size == 2, bool>>
static inline __device__ f4x2_t cast_to_f4_scaled(T x, float scale)
{
    using BaseT = typename scalar_type<T>::type;
    using T8    = typename ck::vector_type<BaseT, 8>::type;
    union
    {
        f4x8_t v8f4;
        f4x2_t f4x2_array[4];
    } ret{0};
    union
    {
        T v_arr[4];
        T8 v8;
    } value{};
    value.v_arr[0] = x;

    ret.v8f4 = cast_to_f4_scaled<T8, stochastic_rounding>(value.v8, scale);
    return ret.f4x2_array[0];
}

template <typename T, bool stochastic_rounding, enable_if_t<scalar_type<T>::vector_size == 8, bool>>
static inline __device__ f4x8_t cast_to_f4_scaled(T x, float scale)
{
    using BaseT = typename scalar_type<T>::type;
    union
    {
        uint32_t bitwise;
        f4x8_t v8f4;
    } value{0};

    if constexpr(stochastic_rounding)
    {
        // use HW clock for stochastic input multiply by incremented thread id
        uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() *
                                                 (get_thread_global_1d_id() + 1));

        if constexpr(is_same_v<BaseT, float>)
            value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_f32(x, rng, scale);
        else if constexpr(is_same_v<BaseT, half_t>)
            value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_f16(x, rng, scale);
        else if constexpr(is_same_v<BaseT, bhalf_t>)
            value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_bf16(x, rng, scale);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(is_same_v<BaseT, float>)
            value.bitwise = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(x, scale);
        else if constexpr(is_same_v<BaseT, half_t>)
            value.bitwise = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f16(x, scale);
        else if constexpr(is_same_v<BaseT, bhalf_t>)
            value.bitwise = __builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16(x, scale);
        else
            static_assert(false_type::value, "Unsupported type.");
    }

    return value.v8f4;
}
#endif
#endif // CK_MX_FP4_CVT_FAST_PATH
} // namespace ck

#endif
