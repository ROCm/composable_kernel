// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "functional4.hpp"
#include "tuple.hpp"

namespace ck {

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_tuple(F&& f, Number<N>)
{
    return unpack([&f](auto&&... xs) { return make_tuple(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_tie(F&& f, Number<N>)
{
    return unpack([&f](auto&&... xs) { return tie(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

// tx and ty are tuple of references, return type of will tuple of referennce (not rvalue)
template <typename... X, typename... Y>
__host__ __device__ constexpr auto concat_tuple_of_reference(const Tuple<X&...>& tx,
                                                             const Tuple<Y&...>& ty)
{
    return unpack2(
        [&](auto&&... zs) { return Tuple<decltype(zs)...>{std::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

template <typename... X, typename... Y>
__host__ __device__ constexpr auto concat_tuple(const Tuple<X...>& tx, const Tuple<Y...>& ty)
{
    return unpack2(
        [&](auto... zs) { return Tuple<decltype(zs)...>{std::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

template <typename... X, typename... Tuples>
__host__ __device__ constexpr auto concat_tuple(const Tuple<X...>& tx, const Tuples&... tuples)
{
    return concat_tuple(tx, concat_tuple(tuples...));
}

namespace detail {

template <typename F, typename X, index_t... Is>
__host__ __device__ constexpr auto transform_tuples_impl(F f, const X& x, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, index_t... Is>
__host__ __device__ constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, typename Z, index_t... Is>
__host__ __device__ constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, const Z& z, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}), z.At(Number<Is>{}))...);
}

} // namespace detail

template <typename F, typename X>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x)
{
    return detail::transform_tuples_impl(
        f, x, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x, const Y& y)
{
    return detail::transform_tuples_impl(
        f, x, y, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y, typename Z>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x, const Y& y, const Z& z)
{
    return detail::transform_tuples_impl(
        f, x, y, z, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename T>
__host__ __device__ constexpr auto UnrollNestedTuple(const T& element)
{
    return make_tuple(element);
}

__host__ __device__ constexpr auto UnrollNestedTuple(const Tuple<>& element) { return element; }

template <typename... Ts>
__host__ __device__ constexpr auto UnrollNestedTuple(const Tuple<Ts...>& tuple)
{
    return unpack([&](auto&&... ts) { return concat_tuple(UnrollNestedTuple(ts)...); }, tuple);
}

} // namespace ck
