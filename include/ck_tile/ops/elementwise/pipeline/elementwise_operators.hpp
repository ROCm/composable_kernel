// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
namespace element_wise {

struct Add
{
    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

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

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const half_t& x1) const
    {
        y = x0 + type_convert<half_t>(x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const float& x1) const
    {
        y = type_convert<half_t>(x0 + x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        y = type_convert<half_t>(x0) + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = x0 + x1;
    };

    /*
    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x1);
        y                  = x0 + x1_tmp;
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x0);
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x1_tmp + x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const float& x0, const bhalf_t& x1) const
    {
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x0 + x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }
    */

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        y = x0 + x1;
    };
};

struct UnarySquare
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        y = x * x;
    };
};

template <typename Op, typename OutputType, typename InputType, typename... InputTypes>
__host__ __device__  
void apply_operation(Op operation, OutputType& output, const tuple<InputType, InputTypes...>& xs) {
    // TODO: If we need to account for nullary operations then this needs a separate overload of
    // apply_operation, due to typing issues with xs.

    if constexpr(sizeof...(InputTypes) == 0)
    {
        // If there is only one input, we can just apply the operation directly
        operation(output, xs.template get<0>());
    }
    else
    {
        // If there are multiple inputs, we need to apply the operation iteratively
        InputType accumulator = xs.template get<0>();

        static_for<0, sizeof...(InputTypes), 1>{}([&](auto i) {
            operation(accumulator, accumulator, xs.get(i));
        });
        output = accumulator;
    }
}

} // namespace element_wise
} // namespace ck_tile
