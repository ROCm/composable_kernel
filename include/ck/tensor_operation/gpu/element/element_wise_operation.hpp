// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

// Need to ensure compiler will fail if there is no matching candidate, instead of compiler
// siliently do implicit type conversion
//
// Method 1:
//
// struct ExampleElementwiseOp
// {
//     template<typename Y, typename X>
//     __host__ __device__ constexpr void
//     operator()(Y&, const X) const;
//
//     template<>
//     __host__ __device__ constexpr void
//     operator()<half_t, half_t>(half_t& y, const half_t& x) const
//     {
//     }
// };
//
// Method 2:
//
// template <typename Y, typename X>
// struct ExampleElementwiseOp;
//
// template <>
// struct ExampleElementwiseOp<float, ck::bhalf_t>
// {
//     __host__ __device__ void operator()(float& y, ck::bhalf_t& x) const
//     {
//     }
// };

struct AddReluAdd
{
    template <typename Y, typename X0, typename X1, typename X2>
    __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&, const X2&) const;

    template <>
    __host__ __device__ constexpr void operator()<half_t, half_t, half_t, half_t>(
        half_t& y, const half_t& x0, const half_t& x1, const half_t& x2) const
    {
        half_t a = x0 + x1;
        half_t b = a > 0 ? a : 0;
        y        = b + x2;
    }

    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float>(float& y,
                                                                              const float& x0,
                                                                              const float& x1,
                                                                              const float& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, float, half_t, half_t>(
        half_t& y, const float& x0, const half_t& x1, const half_t& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }
};

struct AddHardswishAdd
{
    template <typename Y, typename X0, typename X1, typename X2>
    __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&, const X2&) const;

    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float>(float& y,
                                                                              const float& x0,
                                                                              const float& x1,
                                                                              const float& x2) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        float d = c + x2;
        y       = d;
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, half_t, half_t, half_t>(
        half_t& y, const half_t& x0, const half_t& x1, const half_t& x2) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        float d = c + x2;
        y       = d;
    }
};

// C = A * B
// E = FastGelu(C + D0 + D1)
struct AddAddFastGelu
{
    // Fast GeLU
    // https://paperswithcode.com/method/gelu
    // y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    __host__ __device__ static constexpr float GetFastGeLU(float x)
    {
        const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float emu = exp(-u);
        const float cdf = 0.5f + 0.5f * (2.f / (1.f + emu) - 1.f);
        return x * cdf;
    }

    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const
    {
        static_assert(std::is_same_v<std::common_type_t<C, float>, float> &&
                      std::is_same_v<std::common_type_t<D0, float>, float> &&
                      std::is_same_v<std::common_type_t<D1, float>, float>);

        const float y =
            GetFastGeLU(type_convert<float>(c) + type_convert<float>(d0) + type_convert<float>(d1));

        e = type_convert<E>(y);
    }
};

struct Normalize
{
    // FIXME: is double absolutely necessary?
    Normalize(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T>
    __host__ __device__ constexpr void operator()(
        T& y, const T& x, const T& mean, const T& mean_square, const T& gamma, const T& beta) const;

    template <>
    __host__ __device__ constexpr void operator()<float>(float& y,
                                                         const float& x,
                                                         const float& mean,
                                                         const float& mean_square,
                                                         const float& gamma,
                                                         const float& beta) const
    {
        using ck::math::sqrt;

        float variance = mean_square - (mean * mean);
        y = ((x - mean) / sqrt(variance + type_convert<float>(epsilon_))) * gamma + beta;
    };

    template <>
    __host__ __device__ constexpr void operator()<double>(double& y,
                                                          const double& x,
                                                          const double& mean,
                                                          const double& mean_square,
                                                          const double& gamma,
                                                          const double& beta) const
    {
        using ck::math::sqrt;

        double variance = mean_square - (mean * mean);
        y               = ((x - mean) / sqrt(variance + epsilon_)) * gamma + beta;
    };

    // FIXME: is double absolutely necessary?
    double epsilon_;
};

template <typename Y, typename X>
struct UnaryTypeConvert;

template <>
struct UnaryTypeConvert<float, ck::bhalf_t>
{
    __host__ __device__ void operator()(float& y, ck::bhalf_t& x) const
    {
        y = ck::type_convert<float, ck::bhalf_t>(x);
    }
};

template <>
struct UnaryTypeConvert<ck::bhalf_t, float>
{
    __host__ __device__ void operator()(ck::bhalf_t& y, float& x) const
    {
        y = ck::type_convert<ck::bhalf_t, float>(x);
    }
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
