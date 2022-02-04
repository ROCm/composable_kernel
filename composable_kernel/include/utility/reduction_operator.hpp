/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef CK_REDUCTION_OPERATOR_HPP
#define CK_REDUCTION_OPERATOR_HPP

#include "reduction_common.hpp"

namespace ck {

namespace reduce {

// Every binary operator used in reduction is represented by a templated functor class. Each functor
// class must provide at least
// three members:
// 1) GetReductionZeroVal() -- the interface to return the "identity element" for the binary
// operator, "identity element" is the unique
//                    element in the algebraic space that doesn't affect the value of other elements
//                    when operated against them, and the concept is similar to zero vector in
//                    vector space
//                    (http://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/linearalgebra/VectorSpaces.pdf).
// 2) indexable -- boolean value indicating whether indices of the operated elements could be
// recorded. Usually, Min/Max operator could
//                 need to record the indices of elements. For operator like Add/Mul, no need to
//                 record the indices.
// 3) operator() -- the first argument of the operator must be both an input & output, and the
// corresponding variable usually stores
//                  the accumulated result of many operator() calls; the second argument is only an
//                  input. For indexable binary
//                  operator, the second version of operator() has third argument (which is an
//                  output) to indicate whether the
//                  accumulated value (the first argument) has changed, in which case the recorded
//                  accumulated index also need be
//                  changed.

template <class T>
struct Add
{
    using dataType = T;

    __host__ __device__ static constexpr T GetReductionZeroVal() { return static_cast<T>(0.0f); };

    __host__ __device__ inline constexpr void operator()(T& a, T b) const { a = a + b; }
};

template <class T>
struct Mul
{
    using dataType = T;

    __host__ __device__ static constexpr T GetReductionZeroVal() { return static_cast<T>(1.0f); };

    __host__ __device__ inline constexpr void operator()(T& a, T b) const { a = a * b; }
};

template <class T>
struct Max
{
    using dataType = T;

    __host__ __device__ static constexpr T GetReductionZeroVal()
    {
        return NumericLimits<T>::Lowest();
    };

    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        if(a < b)
            a = b;
    }

    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }
};

template <class T>
struct Min
{
    using dataType = T;

    __host__ __device__ static constexpr T GetReductionZeroVal()
    {
        return NumericLimits<T>::Max();
    };

    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        if(a > b)
            a = b;
    }

    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        if(a > b)
        {
            a       = b;
            changed = true;
        }
    }
};

template <class T>
struct AMax
{
    using dataType = T;

    __host__ __device__ static constexpr T GetReductionZeroVal() { return static_cast<T>(0.0f); };

    __host__ __device__ inline constexpr void operator()(T& a, T b) const
    {
        if(a < b)
            a = b;
    }

    __host__ __device__ inline constexpr void operator()(T& a, T b, bool& changed) const
    {
        if(a < b)
        {
            a       = b;
            changed = true;
        }
    }
};

// Unary operators are usually called element-wisely before the reduction is executed on the
// elements.
// They are needed for easy implementation of reduction types of AVG, NRM1, NRM2
template <class T, bool hasDividing>
struct unary_identic
{
    __host__ __device__ unary_identic(const int divider = 1)
    {
        scaler = 1.0f / static_cast<float>(divider);
    };

    __host__ __device__ inline constexpr void operator()(T& y, const T& x) const
    {
        y = x * type_convert<T>(scaler);
    };

    float scaler = 1.0f;
};

template <class T>
struct unary_identic<T, false>
{
    __host__ __device__ unary_identic(const int divider = 1) { (void)divider; };

    __host__ __device__ inline constexpr void operator()(T& y, const T& x) const { y = x; };
};

template <class T, bool hasDividing>
struct unary_square
{
    __host__ __device__ unary_square(const int divider = 1)
    {
        scaler = 1.0f / static_cast<float>(divider);
    };

    __host__ __device__ inline constexpr void operator()(T& y, const T& x) const
    {
        y = x * x;

        y = y * type_convert<T>(scaler);
    };

    float scaler = 1.0f;
};

template <class T>
struct unary_square<T, false>
{
    __host__ __device__ unary_square(const int divider = 1) { (void)divider; };

    __host__ __device__ inline constexpr void operator()(T& y, const T& x) const { y = x * x; };
};

template <class T, bool hasDividing>
struct unary_abs
{
    __host__ __device__ unary_abs(const int divider = 1)
    {
        scaler = 1.0f / static_cast<float>(divider);
    };

    __host__ __device__ inline constexpr void operator()(T& y, const T& x) const
    {
        y = abs(x);

        y = y * type_convert<T>(scaler);
    };

    float scaler = 1.0f;
};

template <class T>
struct unary_abs<T, false>
{
    __host__ __device__ unary_abs(const int divider = 1) { (void)divider; };

    __host__ __device__ inline constexpr void operator()(T& y, const T& x) const { y = abs(x); };
};

template <bool hasDividing>
struct unary_abs<half_t, hasDividing>
{
    __host__ __device__ unary_abs(const int divider = 1)
    {
        scaler = 1.0f / static_cast<float>(divider);
    };

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const
    {
        y = static_cast<half_t>(__habs(x));

        y = y * type_convert<half_t>(scaler);
    };

    float scaler = 1.0f;
};

template <>
struct unary_abs<half_t, false>
{
    __host__ __device__ unary_abs(const int divider = 1) { (void)divider; };

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const
    {
        y = static_cast<half_t>(__habs(x));
    };
};

template <class T>
struct unary_sqrt
{
    __host__ __device__ unary_sqrt(const int divider = 1) { (void)divider; };

    __host__ __device__ inline void operator()(T& y, const T& x) const { y = sqrtf(x); };
};

template <>
struct unary_sqrt<half_t>
{
    __host__ __device__ unary_sqrt(const int divider = 1) { (void)divider; };

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const
    {
        y = static_cast<half_t>(hsqrt(x));
    };
};

}; // end of namespace reduce

// The templated struct reduce_binary_operator maps the enum Ids of binary operators to their
// respective functor classes.
// The "GetReductionZeroVal()" interface and boolean member "indexable" are also provided in
// reduce_binary_operactor for
// easier checking by the upper-layer codes in the kernels.

template <typename T, ReduceTensorOp_t op>
struct reduce_binary_operator;

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::ADD>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    static constexpr bool indexable = false;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::MUL>
{
    using opType   = reduce::Mul<T>;
    using dataType = T;

    static constexpr bool indexable = false;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::MIN>
{
    using opType   = reduce::Min<T>;
    using dataType = T;

    static constexpr bool indexable = true;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::MAX>
{
    using opType   = reduce::Max<T>;
    using dataType = T;

    static constexpr bool indexable = true;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::AMAX>
{
    using opType   = reduce::AMax<T>;
    using dataType = T;

    static constexpr bool indexable = true;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::AVG>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    static constexpr bool indexable = false;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::NORM1>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    static constexpr bool indexable = false;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp_t::NORM2>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    static constexpr bool indexable = false;
};

// The templated struct reduce_unary_operator maps the enum Ids of Reduce operators to two unary
// functor classes.
// The two unary functors are called before and afer the Reduction is executed respectively
template <typename T, ReduceTensorOp_t op, bool isFirsReduce, bool isLastReduce>
struct reduce_unary_operator
{
    using InElementwiseOperation  = reduce::unary_identic<T, false>;
    using AccElementwiseOperation = reduce::unary_identic<T, false>;
};

template <typename T, bool isFirstReduce>
struct reduce_unary_operator<T, ReduceTensorOp_t::AVG, isFirstReduce, true>
{
    using InElementwiseOperation  = reduce::unary_identic<T, false>;
    using AccElementwiseOperation = reduce::unary_identic<T, true>;
};

template <typename T, bool isLastReduce>
struct reduce_unary_operator<T, ReduceTensorOp_t::NORM1, true, isLastReduce>
{
    using InElementwiseOperation  = reduce::unary_abs<T, false>;
    using AccElementwiseOperation = reduce::unary_identic<T, false>;
};

template <typename T, bool isLastReduce>
struct reduce_unary_operator<T, ReduceTensorOp_t::AMAX, true, isLastReduce>
{
    using InElementwiseOperation  = reduce::unary_abs<T, false>;
    using AccElementwiseOperation = reduce::unary_identic<T, false>;
};

template <typename T>
struct reduce_unary_operator<T, ReduceTensorOp_t::NORM2, true, false>
{
    using InElementwiseOperation  = reduce::unary_square<T, false>;
    using AccElementwiseOperation = reduce::unary_identic<T, false>;
};

template <typename T>
struct reduce_unary_operator<T, ReduceTensorOp_t::NORM2, true, true>
{
    using InElementwiseOperation  = reduce::unary_square<T, false>;
    using AccElementwiseOperation = reduce::unary_sqrt<T>;
};

template <typename T>
struct reduce_unary_operator<T, ReduceTensorOp_t::NORM2, false, true>
{
    using InElementwiseOperation  = reduce::unary_identic<T, false>;
    using AccElementwiseOperation = reduce::unary_sqrt<T>;
};

} // end of namespace ck

#endif
