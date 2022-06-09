/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once
#include "data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace binary_element_wise {

template <typename Y, typename X1, typename X2>
struct Add;

template <>
struct Add<double, double, double>
{
    __host__ __device__ constexpr void
    operator()(double& dst, const double& src1, const double& src2) const
    {
        dst = src1 + src2;
    }
};

template <>
struct Add<float, float, float>
{
    __host__ __device__ constexpr void
    operator()(float& dst, const float& src1, const float& src2) const
    {
        dst = src1 + src2;
    }
};

template <>
struct Add<half_t, half_t, half_t>
{
    __host__ __device__ constexpr void
    operator()(half_t& dst, const half_t& src1, const half_t& src2) const
    {
        dst = src1 + src2;
    }
};

template <>
struct Add<bhalf_t, bhalf_t, bhalf_t>
{
    __host__ __device__ constexpr void
    operator()(bhalf_t& dst, const bhalf_t& src1, const bhalf_t& src2) const
    {
        const float x1 = ck::type_convert<float>(src1);
        const float x2 = ck::type_convert<float>(src2);
        const float y  = x1 + x2;
        dst            = ck::type_convert<bhalf_t>(y);
    }
};

template <typename Y, typename X1, typename X2>
struct Substract;

template <>
struct Substract<double, double, double>
{
    __host__ __device__ constexpr void
    operator()(double& dst, const double& src1, const double& src2) const
    {
        dst = src1 - src2;
    }
};

template <>
struct Substract<float, float, float>
{
    __host__ __device__ constexpr void
    operator()(float& dst, const float& src1, const float& src2) const
    {
        dst = src1 - src2;
    }
};

template <>
struct Substract<half_t, half_t, half_t>
{
    __host__ __device__ constexpr void
    operator()(half_t& dst, const half_t& src1, const half_t& src2) const
    {
        dst = src1 - src2;
    }
};

template <>
struct Substract<bhalf_t, bhalf_t, bhalf_t>
{
    __host__ __device__ constexpr void
    operator()(bhalf_t& dst, const bhalf_t& src1, const bhalf_t& src2) const
    {
        const float x1 = ck::type_convert<float>(src1);
        const float x2 = ck::type_convert<float>(src2);
        const float y  = x1 - x2;
        dst            = ck::type_convert<bhalf_t>(y);
    }
};

} // namespace binary_element_wise

namespace element_wise {

struct Add
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = x0 + x1;
    };

    // Question: should half_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = x0 + x1;
    };
};

struct AlphaBetaAdd
{
    AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = alpha_ * x0 + beta_ * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = static_cast<double>(alpha_) * x0 + static_cast<double>(beta_) * x1;
    };

    // Question: should half_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = static_cast<half_t>(alpha_ * static_cast<float>(x0) + beta_ * static_cast<float>(x1));
    };

    float alpha_;
    float beta_;
};

struct AddRelu
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        const float a = x0 + x1;
        y             = a > 0.0f ? a : 0.0f;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        const double a = x0 + x1;
        y              = a > 0.0 ? a : 0.0;
    };

    // Question: should half_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        const half_t a = x0 + x1;
        y              = a > static_cast<half_t>(0.0f) ? a : static_cast<half_t>(0.0f);
    };
};

struct AddHardswish
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > 6.0f ? 6.0f : b) * a * 0.166667f;
        y       = c;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        double a = x0 + x1;
        double b = a + 3.0;
        double c = (b > 0) * (b > 6.0 ? 6.0 : b) * a * 0.166667;
        y        = c;
    };

    // Question: should half_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        float a = x0 + x1;
        float b = a + 3.0f;
        float c = (b > 0) * (b > 6.0f ? 6.0f : b) * a * 0.166667f;
        y       = c;
    };
};

} // namespace element_wise

} // namespace tensor_operation
} // namespace ck
