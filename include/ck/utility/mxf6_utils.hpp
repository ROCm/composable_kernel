// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_CODE_GEN_RTC
#pragma once

#include "ck/utility/numeric_limits.hpp"
#include "ck/utility/mxfp_utils.hpp"

#if CK_MX_ARCH_950 || CK_MX_ARCH_125
#define CK_MX_FP6_CVT_FAST_PATH 1
#else
#define CK_MX_FP6_CVT_FAST_PATH 0
#endif

namespace ck {

namespace utils {
/**
 * @brief Checks if an f6_t value is NaN based on the provided scale.
 *
 * For f6_t data, NaN cannot be represented directly. Instead, this function
 * determines NaN by checking if the scale is set to a quiet NaN.
 *
 * @param scale     The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param dataBytes The f6_t value to check (unused in this implementation).
 * @return true if the scale indicates a NaN value, false otherwise.
 */
template <>
__host__ __device__ inline bool is_nan<f6_t>(e8m0_bexp_t const scale,
                                             f6_t const dataBytes [[maybe_unused]])
{
    // no need to check for data as it does not have NaN representation
    return scale.is_nan();
}

/**
 * @brief Checks if an bf6_t value is NaN based on the provided scale.
 *
 * For bf6_t data, NaN cannot be represented directly. Instead, this function
 * determines NaN by checking if the scale is set to a quiet NaN.
 *
 * @param scale     The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param dataBytes The bf6_t value to check (unused in this implementation).
 * @return true if the scale indicates a NaN value, false otherwise.
 */
template <>
__host__ __device__ inline bool is_nan<bf6_t>(e8m0_bexp_t const scale,
                                              bf6_t const dataBytes [[maybe_unused]])
{
    // no need to check for data as it does not have NaN representation
    return scale.is_nan();
}

/**
 * @brief Checks if an f6_t value is infinite.
 *
 * Because f6_t does not support infinite values, this function always returns false.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param data  The f6_t value to check.
 * @return      Always false, as infinity is not represented in f6_t.
 */
template <>
__host__ __device__ inline bool is_inf<f6_t>(e8m0_bexp_t const scale [[maybe_unused]],
                                             f6_t const data [[maybe_unused]])
{
    // no inf representation for fp6
    return false;
}

/**
 * @brief Checks if an bf6_t value is infinite.
 *
 * Because bf6_t does not support infinite values, this function always returns false.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param data  The bf6_t value to check.
 * @return      Always false, as infinity is not represented in bf6_t.
 */
template <>
__host__ __device__ inline bool is_inf<bf6_t>(e8m0_bexp_t const scale [[maybe_unused]],
                                              bf6_t const data [[maybe_unused]])
{
    // no inf representation for bf6
    return false;
}

/**
 * @brief Checks whether an f6_t value is zero.
 *
 * If the specified f6_t is NaN, this function returns false.
 * Otherwise, it masks out the sign bits and checks if the remaining bits
 * are zero.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param data  The f6_t value to check.
 * @return true if the value is zero; otherwise false.
 */
template <>
__host__ __device__ inline bool is_zero<f6_t>(e8m0_bexp_t const scale, f6_t const data)
{
    if(is_nan<f6_t>(scale, data))
        return false;

    // no need to check for scale as it does not have a 0 representation
    f6_t result = (data & 0b00111111) & NumericUtils<f6_t>::set_sign_mask;

    return result == 0b0;
}

/**
 * @brief Checks whether an bf6_t value is zero.
 *
 * If the specified bf6_t is NaN, this function returns false.
 * Otherwise, it masks out the sign bits and checks if the remaining bits
 * are zero.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param data  The bf6_t value to check.
 * @return true if the value is zero; otherwise false.
 */
template <>
__host__ __device__ inline bool is_zero<bf6_t>(e8m0_bexp_t const scale, bf6_t const data)
{
    if(is_nan<bf6_t>(scale, data))
        return false;

    // no need to check for scale as it does not have a 0 representation
    bf6_t result = (data & 0b00111111) & NumericUtils<bf6_t>::set_sign_mask;

    return result == 0b0;
}

/**
 * @brief Converts an f6_t value to a float based on an e8m0_bexp_t scale factor.
 *
 * Checks if the f6_t value is NaN or zero before performing the conversion.
 * Applies the exponent from the scale to compute the final float result.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param data  The f6_t value to convert.
 * @return      The converted float value.
 */
template <>
__host__ __device__ inline float to_float<f6_t>(e8m0_bexp_t const scale, f6_t const data)
{
    if(is_nan<f6_t>(scale, data))
        return NumericLimits<float>::QuietNaN();

    if(is_zero<f6_t>(scale, data))
        return (data & NumericUtils<f6_t>::negative_zero_mask) ? -0.0f : 0.0f;

    f6_t prepared_data = data & 0b00111111;

    int scale_exp = get_exponent_value<e8m0_bexp_t>(scale);

    return convert_to_float<f6_t>(prepared_data, scale_exp);
}

/**
 * @brief Converts an bf6_t value to a float based on an e8m0_bexp_t scale factor.
 *
 * Checks if the bf6_t value is NaN or zero before performing the conversion.
 * Applies the exponent from the scale to compute the final float result.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param data  The bf6_t value to convert.
 * @return      The converted float value.
 */
template <>
__host__ __device__ inline float to_float<bf6_t>(e8m0_bexp_t const scale, bf6_t const data)
{
    if(is_nan<bf6_t>(scale, data))
        return NumericLimits<float>::QuietNaN();

    if(is_zero<bf6_t>(scale, data))
        return (data & NumericUtils<bf6_t>::negative_zero_mask) ? -0.0f : 0.0f;

    bf6_t prepared_data = data & 0b00111111;

    int scale_exp = get_exponent_value<e8m0_bexp_t>(scale);

    return convert_to_float<bf6_t>(prepared_data, scale_exp);
}

/**
 * @brief Converts a float to f6_t with saturation.
 *
 * If the input is NaN or exceeds the representable range for f6_t, returns
 * the corresponding max normal mask. Handles subnormal cases by returning
 * zero with the appropriate sign.
 *
 * @param value The float value to be converted.
 * @return      The saturated f6_t value.
 */
template <>
__host__ __device__ inline f6_t sat_convert_to_type<f6_t>(float value)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
    {

        return sign ? NumericUtils<f6_t>::data_max_negative_normal_mask
                    : NumericUtils<f6_t>::data_max_positive_normal_mask;
    }

    if(std::abs(value) > NumericLimits<f6_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<f6_t>::data_max_negative_normal_mask
                    : NumericUtils<f6_t>::data_max_positive_normal_mask;

    f6_t res = convert_to_type<f6_t>(value);

    if(std::abs(to_float<f6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<f6_t>::DataMinSubnorm())
        return sign ? NumericUtils<f6_t>::negative_zero_mask
                    : NumericUtils<f6_t>::positive_zero_mask;

    return res;
}

/**
 * @brief Converts a float to bf6_t with saturation.
 *
 * If the input is NaN or exceeds the representable range for bf6_t, returns
 * the corresponding max normal mask. Handles subnormal cases by returning
 * zero with the appropriate sign.
 *
 * @param value The float value to be converted.
 * @return      The saturated bf6_t value.
 */
template <>
__host__ __device__ inline bf6_t sat_convert_to_type<bf6_t>(float value)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
    {

        return sign ? NumericUtils<bf6_t>::data_max_negative_normal_mask
                    : NumericUtils<bf6_t>::data_max_positive_normal_mask;
    }

    if(std::abs(value) > NumericLimits<bf6_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<bf6_t>::data_max_negative_normal_mask
                    : NumericUtils<bf6_t>::data_max_positive_normal_mask;

    bf6_t res = convert_to_type<bf6_t>(value);

    if(std::abs(to_float<bf6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<bf6_t>::DataMinSubnorm())
        return sign ? NumericUtils<bf6_t>::negative_zero_mask
                    : NumericUtils<bf6_t>::positive_zero_mask;

    return res;
}

/**
 * @brief Converts a float to f6_t with saturation and stochastic rounding.
 *
 * If the input is NaN or exceeds the representable range for f6_t, returns
 * the corresponding max normal mask. Handles subnormal cases by returning
 * zero with the appropriate sign.
 *
 * @param value The float value to be converted.
 * @return      The saturated f6_t value.
 */
template <>
__host__ __device__ inline f6_t sat_convert_to_type_sr<f6_t>(float value, uint32_t seed)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
        return sign ? NumericUtils<f6_t>::data_max_negative_normal_mask
                    : NumericUtils<f6_t>::data_max_positive_normal_mask;

    if(std::abs(value) > NumericLimits<f6_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<f6_t>::data_max_negative_normal_mask
                    : NumericUtils<f6_t>::data_max_positive_normal_mask;

    f6_t res = convert_to_type_sr<f6_t>(value, seed);

    if(std::abs(to_float<f6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<f6_t>::DataMinSubnorm())
        return sign ? NumericUtils<f6_t>::negative_zero_mask
                    : NumericUtils<f6_t>::positive_zero_mask;

    return res;
}

/**
 * @brief Converts a float to f6_t with saturation and stochastic rounding.
 *
 * If the input is NaN or exceeds the representable range for f6_t, returns
 * the corresponding max normal mask. Handles subnormal cases by returning
 * zero with the appropriate sign.
 *
 * @param value The float value to be converted.
 * @return      The saturated f6_t value.
 */
template <>
__host__ __device__ inline bf6_t sat_convert_to_type_sr<bf6_t>(float value, uint32_t seed)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
        return sign ? NumericUtils<bf6_t>::data_max_negative_normal_mask
                    : NumericUtils<bf6_t>::data_max_positive_normal_mask;
    if(std::abs(value) > NumericLimits<bf6_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<bf6_t>::data_max_negative_normal_mask
                    : NumericUtils<bf6_t>::data_max_positive_normal_mask;

    bf6_t res = convert_to_type_sr<bf6_t>(value, seed);

    if(std::abs(to_float<bf6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<bf6_t>::DataMinSubnorm())
        return sign ? NumericUtils<bf6_t>::negative_zero_mask
                    : NumericUtils<bf6_t>::positive_zero_mask;

    return res;
}

/* Get packed type from fp6 vector types */
template <typename T>
struct get_f6_packed_type
{
    using type = T;
};

template <>
struct get_f6_packed_type<f6x16_t>
{
    using type = f6x16_pk_t;
};
template <>
struct get_f6_packed_type<f6x16x2_t>
{
    using type = f6x16_pk_t;
};
template <>
struct get_f6_packed_type<f6x32_t>
{
    using type = f6x32_pk_t;
};
template <>
struct get_f6_packed_type<bf6x16x2_t>
{
    using type = bf6x16_pk_t;
};
template <>
struct get_f6_packed_type<bf6x16_t>
{
    using type = bf6x16_pk_t;
};
template <>
struct get_f6_packed_type<bf6x32_t>
{
    using type = bf6x32_pk_t;
};

template <typename T>
using get_f6_packed_type_t = typename get_f6_packed_type<T>::type;

/* Get bit type from fp6 vector types */
template <typename T>
struct get_f6_bit_type
{
    using type             = T;
    static constexpr int N = 1;
};
template <>
struct get_f6_bit_type<f6x16_t>
{
    using type             = f6_t;
    static constexpr int N = 16;
};
template <>
struct get_f6_bit_type<bf6x16_t>
{
    using type             = bf6_t;
    static constexpr int N = 16;
};
template <>
struct get_f6_bit_type<f6x32_t>
{
    using type             = f6_t;
    static constexpr int N = 32;
};
template <>
struct get_f6_bit_type<bf6x32_t>
{
    using type             = bf6_t;
    static constexpr int N = 32;
};

template <typename T>
using get_f6_bit_type_t = typename get_f6_bit_type<T>::type;

/* get fp6/bf6 vector type */
template <index_t N>
struct get_f6_vector_type;
template <>
struct get_f6_vector_type<1>
{
    using type = f6_t;
};
template <>
struct get_f6_vector_type<16>
{
    using type = f6x16_t;
};
template <>
struct get_f6_vector_type<32>
{
    using type = f6x32_t;
};

template <index_t N>
struct get_bf6_vector_type;
template <>
struct get_bf6_vector_type<1>
{
    using type = bf6_t;
};
template <>
struct get_bf6_vector_type<16>
{
    using type = bf6x16_t;
};
template <>
struct get_bf6_vector_type<32>
{
    using type = bf6x32_t;
};

// Result type helper for conversions
template <typename X>
struct f6_result_type
{
    static constexpr int N = scalar_type<X>::vector_size;
    using type             = typename utils::get_f6_vector_type<N>::type;
};

template <typename X>
struct bf6_result_type
{
    static constexpr int N = scalar_type<X>::vector_size;
    using type             = typename utils::get_bf6_vector_type<N>::type;
};

} // namespace utils

#if CK_MX_FP6_CVT_FAST_PATH
// declare
template <typename T, typename T_F6>
inline __device__ enable_if_t<scalar_type<T>::vector_size == 1, T>
cast_from_f6_scaled(T_F6 x, float scale = 1.f);

template <typename T, typename Ts = float, int Opsel = 0, typename T_F6>
inline __device__ enable_if_t<scalar_type<T>::vector_size == 16, T>
cast_from_f6_scaled(T_F6 x, Ts scale = 1.f);

template <typename T, typename T_F6>
inline __device__ enable_if_t<scalar_type<T>::vector_size == 32, T>
cast_from_f6_scaled(T_F6 x, float scale = 1.f);

template <typename T_F6,
          bool stochastic_rounding = false,
          typename T,
          enable_if_t<scalar_type<T>::vector_size == 1, bool> = true>
inline __device__ T_F6 cast_to_f6_scaled(T x, float scale = 1.f);

template <typename T_F6,
          bool stochastic_rounding = false,
          typename T,
          enable_if_t<scalar_type<T>::vector_size == 16, bool> = true>
inline __device__ T_F6 cast_to_f6_scaled(T x, float scale = 1.f);

template <typename T_F6,
          bool stochastic_rounding = false,
          typename T,
          enable_if_t<scalar_type<T>::vector_size == 32, bool> = true>
inline __device__ T_F6 cast_to_f6_scaled(T x, float scale = 1.f);

// definition
#if CK_MX_ARCH_950
// from f6
template <typename T, typename T_F6>
inline __device__ enable_if_t<scalar_type<T>::vector_size == 1, T> cast_from_f6_scaled(T_F6 x,
                                                                                       float scale)
{

    using BaseT         = typename scalar_type<T>::type;
    using f6_vec32_type = conditional_t<is_same_v<T_F6, f6_t>, f6x32_t, bf6x32_t>;
    using T32           = typename vector_type<BaseT, 32>::type;

    utils::get_f6_packed_type_t<f6_vec32_type> f6_packed;
    f6_packed.pack(x, 0);
    union
    {
        T32 vector;
        T array[32];
    } out{};

    out.vector = cast_from_f6_scaled<T32>(f6_vec32_type{f6_packed}, scale);
    return out.array[0];
}

template <typename T, typename Ts, int Opsel, typename T_F6>
inline __device__ enable_if_t<scalar_type<T>::vector_size == 16, T> cast_from_f6_scaled(T_F6 x,
                                                                                        Ts scale)
{
    static_assert(is_same_v<Ts, float>, "Ts must be float");

    using BaseT         = typename scalar_type<T>::type;
    using f6_vec32_type = conditional_t<is_same_v<T_F6, f6x16_t>, f6x32_t, bf6x32_t>;
    using T32           = typename vector_type<BaseT, 32>::type;
    using T_F6X16_PK    = utils::get_f6_packed_type_t<T_F6>;
    using T_F6X32_PK    = utils::get_f6_packed_type_t<f6_vec32_type>;
    constexpr int N     = 32 / scalar_type<T>::vector_size;

    T_F6X32_PK pk;
    const auto& x_packed = x.template AsType<T_F6X16_PK>()[Number<0>{}];
    pk.data_[0]          = x_packed.data_[0];
    pk.data_[1]          = x_packed.data_[1];
    pk.data_[2]          = x_packed.data_[2];

    union
    {
        T32 vector;
        T array[N];
    } out{};

    out.vector = cast_from_f6_scaled<T32>(f6_vec32_type{pk}, scale);
    return out.array[0];
}

template <typename T, typename T_F6>
inline __device__ enable_if_t<scalar_type<T>::vector_size == 32, T> cast_from_f6_scaled(T_F6 x,
                                                                                        float scale)
{
    static_assert(is_same_v<T_F6, f6x32_t> || is_same_v<T_F6, bf6x32_t>,
                  "T_F6 must be either f6x32_t or bf6x32_t");
    using BaseT = typename scalar_type<T>::type;
    if constexpr(is_same_v<T_F6, f6x32_t>)
    {
        if constexpr(is_same_v<BaseT, float>)
            return __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(
                x.template AsType<f6x32_t::data_t>()[Number<0>{}], scale);
        else if constexpr(is_same_v<BaseT, half_t>)
            return __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6(
                x.template AsType<f6x32_t::data_t>()[Number<0>{}], scale);
        else if constexpr(is_same_v<BaseT, bhalf_t>)
            return __builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6(
                x.template AsType<f6x32_t::data_t>()[Number<0>{}], scale);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(is_same_v<BaseT, float>)
            return __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(
                x.template AsType<bf6x32_t::data_t>()[Number<0>{}], scale);
        else if constexpr(is_same_v<BaseT, half_t>)
            return __builtin_amdgcn_cvt_scalef32_pk32_f16_bf6(
                x.template AsType<bf6x32_t::data_t>()[Number<0>{}], scale);
        else if constexpr(is_same_v<BaseT, bhalf_t>)
            return __builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6(
                x.template AsType<bf6x32_t::data_t>()[Number<0>{}], scale);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
}

// to f6
template <typename T_F6,
          bool stochastic_rounding,
          typename T,
          enable_if_t<scalar_type<T>::vector_size == 1, bool>>
inline __device__ T_F6 cast_to_f6_scaled(T x, float scale)
{
    using BaseT         = typename scalar_type<T>::type;
    using f6_vec32_type = conditional_t<is_same_v<T_F6, f6_t>, f6x32_t, bf6x32_t>;
    using T32           = typename vector_type<BaseT, 32>::type;
    union
    {
        T32 vector;
        T array[32];
    } in{x};

    auto f6_vector = cast_to_f6_scaled<f6_vec32_type, stochastic_rounding>(in.vector, scale);
    auto f6_packed = static_cast<utils::get_f6_packed_type_t<f6_vec32_type>>(f6_vector);

    return f6_packed.unpack(0);
}

template <typename T_F6,
          bool stochastic_rounding,
          typename T,
          enable_if_t<scalar_type<T>::vector_size == 16, bool>>
inline __device__ T_F6 cast_to_f6_scaled(T x, float scale)
{
    using BaseT         = typename scalar_type<T>::type;
    using f6_vec32_type = conditional_t<is_same_v<T_F6, f6x16_t>, f6x32_t, bf6x32_t>;
    using T32           = typename vector_type<BaseT, 32>::type;
    constexpr int N     = 32 / scalar_type<T>::vector_size;
    using T6X16_TYPE    = utils::get_f6_packed_type_t<T_F6>;
    using T6X32_TYPE    = utils::get_f6_packed_type_t<f6_vec32_type>;
    union
    {
        T array[N];
        T32 vector;
    } in{{x, x}};

    auto f6_vector = cast_to_f6_scaled<f6_vec32_type, stochastic_rounding>(in.vector, scale);
    const T6X32_TYPE& pk_f6_vector = f6_vector.template AsType<T6X32_TYPE>()[Number<0>{}];
    T6X16_TYPE pk_out;
    pk_out.data_[0] = pk_f6_vector.data_[0];
    pk_out.data_[1] = pk_f6_vector.data_[1];
    pk_out.data_[2] = pk_f6_vector.data_[2];

    return T_F6{pk_out};
}

template <typename T_F6,
          bool stochastic_rounding,
          typename T,
          enable_if_t<scalar_type<T>::vector_size == 32, bool>>
inline __device__ T_F6 cast_to_f6_scaled(T x, float scale)
{
    static_assert(is_same_v<T_F6, f6x32_t> || is_same_v<T_F6, bf6x32_t>,
                  "T_F6 must be either f6x32_t or bf6x32_t");
    using BaseT = typename scalar_type<T>::type;

    if constexpr(stochastic_rounding)
    {
        uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() *
                                                 (get_thread_global_1d_id() + 1));

        if constexpr(is_same_v<T_F6, f6x32_t>)
        {
            if constexpr(is_same_v<BaseT, float>)
                return f6x32_t{__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32(x, rng, scale)};
            else if constexpr(is_same_v<BaseT, half_t>)
                return f6x32_t{__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f16(x, rng, scale)};
            else if constexpr(is_same_v<BaseT, bhalf_t>)
                return f6x32_t{__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_bf16(x, rng, scale)};
            else
                static_assert(false_type::value, "Unsupported type.");
        }
        else
        {
            if constexpr(is_same_v<BaseT, float>)
                return bf6x32_t{__builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f32(x, rng, scale)};
            else if constexpr(is_same_v<BaseT, half_t>)
                return bf6x32_t{__builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f16(x, rng, scale)};
            else if constexpr(is_same_v<BaseT, bhalf_t>)
                return bf6x32_t{__builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_bf16(x, rng, scale)};
            else
                static_assert(false_type::value, "Unsupported type.");
        }
    }
    else
    {
        if constexpr(is_same_v<T_F6, f6x32_t>)
        {
            if constexpr(is_same_v<BaseT, float>)
            {
                float16_t even, odd;
                float* src      = reinterpret_cast<float*>(&x);
                float* even_ptr = reinterpret_cast<float*>(&even);
                float* odd_ptr  = reinterpret_cast<float*>(&odd);

                static_for<0, 16, 1>{}([&](auto i) {
                    even_ptr[i] = src[2 * i];
                    odd_ptr[i]  = src[2 * i + 1];
                });

                /* first and second src inputs are interleaved in the packed result. */
                return f6x32_t{__builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(even, odd, scale)};
            }
            else if constexpr(is_same_v<BaseT, half_t>)
                return f6x32_t{__builtin_amdgcn_cvt_scalef32_pk32_fp6_f16(x, scale)};
            else if constexpr(is_same_v<BaseT, bhalf_t>)
                return f6x32_t{__builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16(x, scale)};
            else
                static_assert(false_type::value, "Unsupported type.");
        }
        else
        {
            if constexpr(is_same_v<BaseT, float>)
            {
                float16_t even, odd;
                float* src      = reinterpret_cast<float*>(&x);
                float* even_ptr = reinterpret_cast<float*>(&even);
                float* odd_ptr  = reinterpret_cast<float*>(&odd);

                static_for<0, 16, 1>{}([&](auto i) {
                    even_ptr[i] = src[2 * i];
                    odd_ptr[i]  = src[2 * i + 1];
                });
                /* first and second src inputs are interleaved in the packed result. */
                return bf6x32_t{__builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(even, odd, scale)};
            }
            else if constexpr(is_same_v<BaseT, half_t>)
                return bf6x32_t{__builtin_amdgcn_cvt_scalef32_pk32_bf6_f16(x, scale)};
            else if constexpr(is_same_v<BaseT, bhalf_t>)
                return bf6x32_t{__builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16(x, scale)};
            else
                static_assert(false_type::value, "Unsupported type.");
        }
    }
}

#elif CK_MX_ARCH_125
// from f6
template <typename T, typename T_F6>
inline __device__ enable_if_t<scalar_type<T>::vector_size == 1, T> cast_from_f6_scaled(T_F6 x,
                                                                                       float scale)
{
    using BaseT         = typename scalar_type<T>::type;
    using f6_vec16_type = conditional_t<is_same_v<T_F6, f6_t>, f6x16_t, bf6x16_t>;
    using T16           = typename vector_type<BaseT, 16>::type;

    utils::get_f6_packed_type_t<f6_vec16_type> f6_packed;
    f6_packed.pack(x, 0);
    f6_vec16_type f6_vector{f6_packed};
    union
    {
        T16 vector;
        T array[16];
    } out{};

    out.vector = cast_from_f6_scaled<T16>(f6_vector, scale);
    return out.array[0];
}

template <typename T, typename Ts, int Opsel, typename T_F6>
inline __device__ enable_if_t<scalar_type<T>::vector_size == 16, T> cast_from_f6_scaled(T_F6 x,
                                                                                        Ts scale)
{
    static_assert(is_same_v<T_F6, f6x16_t> || is_same_v<T_F6, bf6x16_t>,
                  "T_F6 must be either f6x16_t or bf6x16_t");
    static_assert(sizeof(Ts) == 4, "Ts must be float or uint32_t");

    using BaseT     = typename scalar_type<T>::type;
    uint32_t scale4 = (is_same_v<Ts, float>)
                          ? bit_cast<uint32_t>(utils::get_exponent_value(e8m0_bexp_t(scale)))
                          : bit_cast<uint32_t>(scale);

    if constexpr(is_same_v<T_F6, f6x16_t>)
    {
        if constexpr(is_same_v<BaseT, float>)
            return __builtin_amdgcn_cvt_scale_pk16_f32_fp6(
                x.template AsType<f6x16_t::data_t>()[Number<0>{}], scale4, Opsel);
        else if constexpr(is_same_v<BaseT, half_t>)
            return __builtin_amdgcn_cvt_scale_pk16_f16_fp6(
                x.template AsType<f6x16_t::data_t>()[Number<0>{}], scale4, Opsel);
        else if constexpr(is_same_v<BaseT, bhalf_t>)
            return __builtin_amdgcn_cvt_scale_pk16_bf16_fp6(
                x.template AsType<f6x16_t::data_t>()[Number<0>{}], scale4, Opsel);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
    else
    {
        if constexpr(is_same_v<BaseT, float>)
            return __builtin_amdgcn_cvt_scale_pk16_f32_bf6(
                x.template AsType<bf6x16_t::data_t>()[Number<0>{}], scale4, Opsel);
        else if constexpr(is_same_v<BaseT, half_t>)
            return __builtin_amdgcn_cvt_scale_pk16_f16_bf6(
                x.template AsType<bf6x16_t::data_t>()[Number<0>{}], scale4, Opsel);
        else if constexpr(is_same_v<BaseT, bhalf_t>)
            return __builtin_amdgcn_cvt_scale_pk16_bf16_bf6(
                x.template AsType<bf6x16_t::data_t>()[Number<0>{}], scale4, Opsel);
        else
            static_assert(false_type::value, "Unsupported type.");
    }
}

template <typename T, typename T_F6>
inline __device__ enable_if_t<scalar_type<T>::vector_size == 32, T> cast_from_f6_scaled(T_F6 x,
                                                                                        float scale)
{
    using f6_vec16_type = conditional_t<is_same_v<T_F6, f6x32_t>, f6x16_t, bf6x16_t>;
    using BaseT         = typename scalar_type<T>::type;
    using T16           = typename vector_type<BaseT, 16>::type;
    using T6X16_TYPE    = utils::get_f6_packed_type_t<f6_vec16_type>;
    using T6X32_TYPE    = utils::get_f6_packed_type_t<T_F6>;
    union
    {
        T16 array[2];
        T vector;
    } out{};

    // Extract the f6x32_pk_t from x
    const T6X32_TYPE& x_pk32 = x.template AsType<T6X32_TYPE>()[Number<0>{}];

    // Manually split: f6x32_pk_t has data_[6], split into two f6x16_pk_t with data_[3] each
    T6X16_TYPE pk_lo, pk_hi;
    pk_lo.data_[0] = x_pk32.data_[0];
    pk_lo.data_[1] = x_pk32.data_[1];
    pk_lo.data_[2] = x_pk32.data_[2];
    pk_hi.data_[0] = x_pk32.data_[3];
    pk_hi.data_[1] = x_pk32.data_[4];
    pk_hi.data_[2] = x_pk32.data_[5];

    // Wrap into f6x16_t by constructing from packed types
    f6_vec16_type f6_lo{pk_lo};
    f6_vec16_type f6_hi{pk_hi};

    out.array[0] = cast_from_f6_scaled<T16>(f6_lo, scale);
    out.array[1] = cast_from_f6_scaled<T16>(f6_hi, scale);
    return out.vector;
}

// to f6
template <typename T_F6,
          bool stochastic_rounding,
          typename T,
          enable_if_t<scalar_type<T>::vector_size == 1, bool>>
inline __device__ T_F6 cast_to_f6_scaled(T x, float scale)
{
    using BaseT         = typename scalar_type<T>::type;
    using f6_vec16_type = conditional_t<is_same_v<T_F6, f6_t>, f6x16_t, bf6x16_t>;
    using T16           = typename vector_type<BaseT, 16>::type;
    union
    {
        T16 vector;
        T array[16];
    } in{};
    in.array[0] = x;

    auto f6_vector = cast_to_f6_scaled<f6_vec16_type, stochastic_rounding>(in.vector, scale);
    auto f6_packed = static_cast<utils::get_f6_packed_type_t<f6_vec16_type>>(f6_vector);

    return f6_packed.unpack(0);
}

template <typename T_F6,
          bool stochastic_rounding,
          typename T,
          enable_if_t<scalar_type<T>::vector_size == 16, bool>>
inline __device__ T_F6 cast_to_f6_scaled(T x, float scale)
{
    static_assert(is_same_v<T_F6, f6x16_t> || is_same_v<T_F6, bf6x16_t>,
                  "T_F6 must be either f6x16_t or bf6x16_t");
    using BaseT = typename scalar_type<T>::type;

    if constexpr(stochastic_rounding)
    {
        // use HW clock for stochastic input multiply by incremented thread id
        uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_readcyclecounter() *
                                                 (get_thread_global_1d_id() + 1));

        if constexpr(is_same_v<T_F6, f6x16_t>)
        {
            if constexpr(is_same_v<BaseT, float>)
                return f6x16_t{__builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_f32(x, rng, scale)};
            else if constexpr(is_same_v<BaseT, half_t>)
                return f6x16_t{__builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_f16(x, rng, scale)};
            else if constexpr(is_same_v<BaseT, bhalf_t>)
                return f6x16_t{__builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_bf16(x, rng, scale)};
            else
                static_assert(false_type::value, "Unsupported type.");
        }
        else
        {
            if constexpr(is_same_v<BaseT, float>)
                return bf6x16_t{__builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_f32(x, rng, scale)};
            else if constexpr(is_same_v<BaseT, half_t>)
                return bf6x16_t{__builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_f16(x, rng, scale)};
            else if constexpr(is_same_v<BaseT, bhalf_t>)
                return bf6x16_t{__builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_bf16(x, rng, scale)};
            else
                static_assert(false_type::value, "Unsupported type.");
        }
    }
    else
    {
        if constexpr(is_same_v<T_F6, f6x16_t>)
        {
            if constexpr(is_same_v<BaseT, float>)
                return f6x16_t{__builtin_amdgcn_cvt_scalef32_pk16_fp6_f32(x, scale)};
            else if constexpr(is_same_v<BaseT, half_t>)
                return f6x16_t{__builtin_amdgcn_cvt_scalef32_pk16_fp6_f16(x, scale)};
            else if constexpr(is_same_v<BaseT, bhalf_t>)
                return f6x16_t{__builtin_amdgcn_cvt_scalef32_pk16_fp6_bf16(x, scale)};
            else
                static_assert(false_type::value, "Unsupported type.");
        }
        else
        {
            if constexpr(is_same_v<BaseT, float>)
                return bf6x16_t{__builtin_amdgcn_cvt_scalef32_pk16_bf6_f32(x, scale)};
            else if constexpr(is_same_v<BaseT, half_t>)
                return bf6x16_t{__builtin_amdgcn_cvt_scalef32_pk16_bf6_f16(x, scale)};
            else if constexpr(is_same_v<BaseT, bhalf_t>)
                return bf6x16_t{__builtin_amdgcn_cvt_scalef32_pk16_bf6_bf16(x, scale)};
            else
                static_assert(false_type::value, "Unsupported type.");
        }
    }
}

template <typename T_F6,
          bool stochastic_rounding,
          typename T,
          enable_if_t<scalar_type<T>::vector_size == 32, bool>>
inline __device__ T_F6 cast_to_f6_scaled(T x, float scale)
{
    using BaseT         = typename scalar_type<T>::type;
    using f6_vec16_type = conditional_t<is_same_v<T_F6, f6x32_t>, f6x16_t, bf6x16_t>;
    using T16           = typename vector_type<BaseT, 16>::type;
    using T6X16_TYPE    = utils::get_f6_packed_type_t<f6_vec16_type>;
    using T6X32_TYPE    = utils::get_f6_packed_type_t<T_F6>;
    union
    {
        T vector;
        T16 array[2];
    } in{x};

    // Convert each half to f6x16_t
    f6_vec16_type f6_lo = cast_to_f6_scaled<f6_vec16_type, stochastic_rounding>(in.array[0], scale);
    f6_vec16_type f6_hi = cast_to_f6_scaled<f6_vec16_type, stochastic_rounding>(in.array[1], scale);

    // Extract packed types from wrappers
    const T6X16_TYPE& pk_lo = f6_lo.template AsType<T6X16_TYPE>()[Number<0>{}];
    const T6X16_TYPE& pk_hi = f6_hi.template AsType<T6X16_TYPE>()[Number<0>{}];

    // Manually combine: two f6x16_pk_t with data_[3] each into f6x32_pk_t with data_[6]
    T6X32_TYPE pk_out;
    pk_out.data_[0] = pk_lo.data_[0];
    pk_out.data_[1] = pk_lo.data_[1];
    pk_out.data_[2] = pk_lo.data_[2];
    pk_out.data_[3] = pk_hi.data_[0];
    pk_out.data_[4] = pk_hi.data_[1];
    pk_out.data_[5] = pk_hi.data_[2];

    // Wrap into f6x32_t and return
    return T_F6{pk_out};
}
#endif
#endif // CK_MX_FP4_CVT_FAST_PATH

} // namespace ck
#endif
