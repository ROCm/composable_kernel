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
} // namespace tensor_operation
} // namespace ck
