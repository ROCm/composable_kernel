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
#ifndef CK_REDUCTION_OPERATOR_MAPPING_HPP
#define CK_REDUCTION_OPERATOR_MAPPING_HPP

#include "reduction_operator.hpp"
#include "reduction_enums.hpp"
#include "element_wise_operation.hpp"

namespace ck {

// The templated struct reduce_binary_operator maps the enum Ids of binary operators to their
// respective functor classes.
// The boolean member "indexable" are also provided in reduce_binary_operactor for
// easier checking by the upper-layer codes in the kernels.

template <typename T, ReduceTensorOp Op>
struct reduce_binary_operator;

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp::ADD>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    static constexpr bool indexable = false;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp::MUL>
{
    using opType   = reduce::Mul<T>;
    using dataType = T;

    static constexpr bool indexable = false;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp::MIN>
{
    using opType   = reduce::Min<T>;
    using dataType = T;

    static constexpr bool indexable = true;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp::MAX>
{
    using opType   = reduce::Max<T>;
    using dataType = T;

    static constexpr bool indexable = true;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp::AMAX>
{
    using opType   = reduce::AMax<T>;
    using dataType = T;

    static constexpr bool indexable = true;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp::AVG>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    static constexpr bool indexable = false;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp::NORM1>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    static constexpr bool indexable = false;
};

template <typename T>
struct reduce_binary_operator<T, ReduceTensorOp::NORM2>
{
    using opType   = reduce::Add<T>;
    using dataType = T;

    static constexpr bool indexable = false;
};

// The templated struct reduce_unary_operator maps the enum Ids of Reduce operators to two unary
// functor classes.
// The two unary functors are called before and afer the Reduction is executed respectively
template <typename T, ReduceTensorOp Op, bool IsFirstReduce, bool IsLastReduce>
struct reduce_unary_operator
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnaryIdentic<T, T>;
    using AccElementwiseOperation = tensor_operation::element_wise::UnaryIdentic<T, T>;
};

template <typename T, bool IsFirstReduce>
struct reduce_unary_operator<T, ReduceTensorOp::AVG, IsFirstReduce, true>
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnaryIdentic<T, T>;
    using AccElementwiseOperation = tensor_operation::element_wise::UnaryIdentic<T, T, true>;
};

template <typename T, bool IsLastReduce>
struct reduce_unary_operator<T, ReduceTensorOp::NORM1, true, IsLastReduce>
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnaryAbs<T, T>;
    using AccElementwiseOperation = tensor_operation::element_wise::UnaryIdentic<T, T>;
};

template <typename T, bool IsLastReduce>
struct reduce_unary_operator<T, ReduceTensorOp::AMAX, true, IsLastReduce>
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnaryAbs<T, T>;
    using AccElementwiseOperation = tensor_operation::element_wise::UnaryIdentic<T, T>;
};

template <typename T>
struct reduce_unary_operator<T, ReduceTensorOp::NORM2, true, false>
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnarySquare<T, T>;
    using AccElementwiseOperation = tensor_operation::element_wise::UnaryIdentic<T, T>;
};

template <typename T>
struct reduce_unary_operator<T, ReduceTensorOp::NORM2, true, true>
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnarySquare<T, T>;
    using AccElementwiseOperation = tensor_operation::element_wise::UnarySqrt<T, T>;
};

template <typename T>
struct reduce_unary_operator<T, ReduceTensorOp::NORM2, false, true>
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnaryIdentic<T, T>;
    using AccElementwiseOperation = tensor_operation::element_wise::UnarySqrt<T, T>;
};

} // end of namespace ck

#endif
