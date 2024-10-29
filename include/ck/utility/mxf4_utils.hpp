// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/mxfp_utils.hpp"

namespace ck {

__host__ inline int clz(uint32_t x) { return __builtin_clz(x); }
__device__ inline int clz(uint32_t x) { return __clz(x); }

} // namespace ck

namespace ck::utils {

// template <typename X, typename Y, bool negative_zero_nan, bool clip, bool stoch>
// __host__ __device__ Y cast_to_f8(X x, uint32_t rng)
// {
//     // check datatypes
//     constexpr bool is_half  = std::is_same<X, half_t>::value;
//     constexpr bool is_float = std::is_same<X, float>::value;
//     static_assert(is_half || is_float, "Only half and float can be casted.");

//     return run_cast_to_f8<X, Y, negative_zero_nan, clip, stoch>(x, rng);
// }

// template <typename X, typename Y, bool negative_zero_nan>
// __host__ __device__ Y cast_from_f8(X x)
// {
//     // check datatype
//     constexpr bool is_half  = std::is_same<Y, half_t>::value;
//     constexpr bool is_float = std::is_same<Y, float>::value;
//     static_assert(is_half || is_float, "only half and float are supported.");

//     return run_cast_from_f8<X, Y, negative_zero_nan>(x);
// }

template <>
__host__ __device__ inline bool is_nan<f4_t>(e8m0_scale_t const scale,
                                             f4_t const dataBytes [[maybe_unused]])
{
    // no need to check for data as it does not have NaN representation
    return scale == NumericLimits<e8m0_scale_t>::QuietNaN();
}

// no infinity representation in ocp_e2m1_mxfp4 will always return false
template <>
__host__ __device__ inline bool is_inf<f4_t>(e8m0_scale_t const scale [[maybe_unused]],
                                             f4_t const data [[maybe_unused]])
{
    // no inf representation for ocp_e2m1_mxfp4
    return false;
}

template <>
__host__ __device__ inline bool is_zero<f4_t>(e8m0_scale_t const scale, f4_t const data)
{
    if(is_nan<f4_t>(scale, data))
        return false;

    // no need to check for scale as it does not have a 0 representation
    f4_t data = (data & 0b00001111) & NumericUtils<e8m0_scale_t>::set_sign_mask;

    return data == 0b0;
}

template <>
__host__ __device__ inline float to_float<f4_t>(e8m0_scale_t const scale, f4_t const data)
{
    if(is_nan<f4_t>(scale, data))
        return std::numeric_limits<float>::quiet_NaN();

    if(is_zero<f4_t>(scale, data))
        return 0.0f;

    uint8_t data = data & 0b00001111;

    int scale_exp = get_exponent_value<e8m0_scale_t>(scale);

    return convert_to_float<f4_t>(data, scale_exp);
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

    if(std::abs(value) > NumericLimits<f4_t>::Max()) // covers inf case as well
        return sign ? NumericUtils<f4_t>::data_max_negative_normal_mask
                    : NumericUtils<f4_t>::data_max_positive_normal_mask;

    f4_t res = convert_to_type<f4_t>(value);

    if(std::abs(to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(), res)) <
       NumericUtils<f4_t>::DataMinSubnorm())
        return value < 0 ? NumericUtils<f4_t>::negative_zero_mask
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

    if(std::abs(value) > NumericLimits<f4_t>::Max()) // covers inf case as well
        return sign ? NumericUtils<f4_t>::data_max_negative_normal_mask
                    : NumericUtils<f4_t>::data_max_positive_normal_mask;

    f4_t res = convert_to_type_sr<f4_t>(value, seed);

    if(std::abs(to_float<f4_t>(NumericLimits<e8m0_scale_t>::Binary_1(), res)) <
       NumericUtils<f4_t>::DataMinSubnorm())
        return value < 0 ? NumericUtils<f4_t>::negative_zero_mask
                         : NumericUtils<f4_t>::positive_zero_mask;

    return res;
}

} // namespace ck::utils
