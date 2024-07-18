// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "functional2.hpp"
#include "sequence.hpp"
#include "tuple.hpp"

namespace ck {

namespace detail {
template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<Tuple<Xs...>, Tuple<Ys...>>
{
    using type = Tuple<Xs..., Ys...>;
};

template <typename T, index_t N>
struct StaticallyIndexedArrayImpl
{
    using type =
        typename tuple_concat<typename StaticallyIndexedArrayImpl<T, N / 2>::type,
                              typename StaticallyIndexedArrayImpl<T, N - N / 2>::type>::type;
};

template <typename T>
struct StaticallyIndexedArrayImpl<T, 0>
{
    using type = Tuple<>;
};

template <typename T>
struct StaticallyIndexedArrayImpl<T, 1>
{
    using type = Tuple<T>;
};
} // namespace detail

template <typename T, index_t N>
using StaticallyIndexedArray = typename detail::StaticallyIndexedArrayImpl<T, N>::type;

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_statically_indexed_array(const X& x, const Xs&... xs)
{
    return StaticallyIndexedArray<X, sizeof...(Xs) + 1>(x, static_cast<X>(xs)...);
}

// make empty StaticallyIndexedArray
template <typename X>
__host__ __device__ constexpr auto make_statically_indexed_array()
{
    return StaticallyIndexedArray<X, 0>();
}

} // namespace ck
