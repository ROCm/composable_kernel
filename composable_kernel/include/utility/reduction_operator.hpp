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

#include "common_header.hpp"

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
template <class T, bool hasDividing=false>
struct unary_identic
{
    __host__ __device__ unary_identic(const int divider = 1) { (void)divider; };

    __host__ __device__ inline constexpr void operator()(T& y, const T& x) const { y = x; };
};

template <class T>
struct unary_identic<T, true>
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

template <class T, bool hasDividing=false>
struct unary_square
{
    __host__ __device__ unary_square(const int divider = 1) { (void)divider; };

    __host__ __device__ inline constexpr void operator()(T& y, const T& x) const { y = x * x; };
};


template <class T>
struct unary_square<T, true>
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

static inline __device__ half_t abs(half_t x) { return __habs(x); };
static inline __device__ half_t sqrtf(half_t x) { return hsqrt(x); };

template <class T, bool hasDividing=false>
struct unary_abs
{
    __host__ __device__ unary_abs(const int divider = 1) { (void)divider; };

    __host__ __device__ inline constexpr void operator()(T& y, const T& x) const { y = abs(x); };
};

template <class T>
struct unary_abs<T, true>
{
    __host__ __device__ unary_abs(const int divider = 1)
    {
        scaler = 1.0f / static_cast<float>(divider);
    };

    __host__ __device__ inline constexpr void operator()(T& y, const T& x) const
    {
        y = x * type_convert<T>(scaler);
        y = abs(y);
    };

    float scaler = 1.0f;
};

template <class T>
struct unary_sqrt
{
    __host__ __device__ unary_sqrt(const int divider = 1) { (void)divider; };

    __host__ __device__ inline void operator()(T& y, const T& x) const { y = sqrtf(x); };
};

}; // end of namespace reduce

} // end of namespace ck

#endif
