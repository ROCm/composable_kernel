// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math_v2.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct PassThrough
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<double, double>(double& y, const double& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<int32_t, int32_t>(int32_t& y, const int32_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, float>(bhalf_t& y, const float& x) const
    {
        y = type_convert<bhalf_t>(x);
    }

    template <>
    __host__ __device__ void operator()<int8_t, int8_t>(int8_t& y, const int8_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<int8_t, int32_t>(int8_t& y, const int32_t& x) const
    {
        y = type_convert<int8_t>(x);
    }
};

struct UnaryConvert
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x);
    }
};

struct Scale
{
    __host__ __device__ Scale(float scale) : scale_(scale) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = scale_ * x;
    };

    float scale_;
};

struct UnaryDivide
{
    __host__ __device__ UnaryDivide(const int32_t divider = 1) : divider_(divider) {}

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = x / type_convert<T>(divider_);
    };

    int32_t divider_ = 1;
};

struct UnarySquare
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        y = x * x;
    };
};

struct UnaryAbs
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::abs(x);
    };
};

struct UnarySqrt
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::sqrt(x);
    };
};

struct Relu
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        y = x > 0 ? x : 0;
    }

    template <>
    __host__ __device__ void operator()(bhalf_t& y, const bhalf_t& x) const
    {
        float x_f32 = ck::type_convert<float>(x);
        float y_f32 = x_f32 > 0 ? x_f32 : 0;
        y           = ck::type_convert<bhalf_t>(y_f32);
    }
};

// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
struct FastGelu
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        const float u   = float(2) * x * (float(0.035677) * x * x + float(0.797885));
        const float emu = exp(-u);
        const float cdf = float(0.5) + float(0.5) * (float(2) / (float(1) + emu) - float(1));

        y = x * cdf;
    }
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
