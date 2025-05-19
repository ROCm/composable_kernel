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

// template <typename Op, typename AccumulatorType, typename TermType, typename... Args>  
// __host__ __device__  
// AccumulatorType n_ary_operation(Op operation, TermType first_arg, Args... rest_args) { 
//     AccumulatorType accumulator = first_arg;  
  
//     AccumulatorType dummy[] = {0, ( (void)(accumulator = operation(accumulator, rest_args, accumulator)), 0 )... };  
//     (void)dummy; // Suppress unused variable warning for dummy array  
 
//     // The initial 0 in `int dummy[] = {0, ...}` handles the case where rest_args is empty.  
//     // If rest_args is empty, dummy becomes {0}.  
//     // If rest_args is not empty, dummy becomes {0, result_of_pack_expansion_1, result_of_pack_expansion_2, ...}  
  
//     return accumulator;  
// }  

template <typename Op, typename AccumulatorType, typename TermType, typename... Args>  
__host__ __device__  
void n_ary_operation2(Op operation, AccumulatorType& output, TermType first_arg, Args... rest_args) { 
    // AccumulatorType accumulator = first_arg;
    // accumulator = first_arg ;
    // using ValueType = typename std::remove_reference<AccumulatorType>::type;

    TermType accumulator = 0;
    // ValueType zero = ValueType{}; // <- null pointer, but I would like a value of '0' at that address, not a null pointer
    AccumulatorType dummy[] = {AccumulatorType{0}, ( (void)(operation(accumulator, rest_args, accumulator)), AccumulatorType{0})... };  // Use extra register here? We should probably get rid of the assignment or will the compiler workout we don't need this variable
    (void)dummy; // Suppress unused variable warning for dummy array 
    operation(output, first_arg, accumulator); // This is the final result of the operation. 
}  
  
// template <typename Op, typename AccumulatorType, typename TermType, typename... Args>  
// __host__ __device__  
// AccumulatorType op_variadic_fold(Op operation, TermType first, Args... rest) {  
//     return n_ary_operation(operation, first, rest...);
// } 

} // namespace element_wise
} // namespace ck_tile
