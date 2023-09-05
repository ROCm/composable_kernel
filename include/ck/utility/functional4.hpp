// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "sequence.hpp"
#include "tuple.hpp"
#include "array.hpp"

namespace ck {

namespace detail {

template <typename Indices>
struct unpack_impl;

template <index_t... Is>
struct unpack_impl<Sequence<Is...>>
{
    template <typename F, typename X>
    __host__ __device__ constexpr auto operator()(F&& f, X&& x) const
    {
#if 0
        return std::forward<F>(f)(std::forward<X>(x).At(Number<Is>{})...);
#else
        return std::forward<F>(f)(std::forward<X>(x).template At<Is>()...);
#endif
    }
};

template <typename Seq0, typename Seq1>
struct unpack2_impl;

// TODO: remove this, after properly implementing unpack that takes any number of containers
template <index_t... Is, index_t... Js>
struct unpack2_impl<Sequence<Is...>, Sequence<Js...>>
{
    template <typename F, typename X, typename Y>
    __host__ __device__ constexpr auto operator()(F&& f, X&& x, Y&& y) const
    {
#if 0
        return std::forward<F>(f)(std::forward<X>(x).At(Number<Is>{})...,
                                  std::forward<Y>(y).At(Number<Js>{})...);
#else
        return std::forward<F>(f)(std::forward<X>(x).template At<Is>()...,
                                  std::forward<Y>(y).template At<Js>()...);
#endif
    }
};

} // namespace detail

template <typename F, typename X>
__host__ __device__ constexpr auto unpack(F&& f, X&& x)
{
    using X_ = remove_reference_t<X>;
    return detail::unpack_impl<typename arithmetic_sequence_gen<0, X_::Size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x));
}

// TODO: properly implement unpack that takes any number of containers
template <typename F, typename X, typename Y>
__host__ __device__ constexpr auto unpack2(F&& f, X&& x, Y&& y)
{
    using X_ = remove_reference_t<X>;
    using Y_ = remove_reference_t<Y>;
    return detail::unpack2_impl<typename arithmetic_sequence_gen<0, X_::Size(), 1>::type,
                                typename arithmetic_sequence_gen<0, Y_::Size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x), std::forward<Y>(y));
}

} // namespace ck
