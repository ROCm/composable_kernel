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


template <typename Op, typename AccumulatorType, typename TermType, typename... Args>  
__host__ __device__  
void binary_operation(Op operation, AccumulatorType& output, TermType first_arg, Args... rest_args) { 

    TermType accumulator = 0;

    AccumulatorType dummy[] = {AccumulatorType{0}, ( (void)(operation(accumulator, rest_args, accumulator)), AccumulatorType{0})... };  
    (void)dummy; // Suppress unused variable warning for dummy array 
    operation(output, first_arg, accumulator); // This is the final result of the operation. 
}  

// TODO: implement a generic operation for unitary functions???

} // namespace element_wise
} // namespace ck_tile
