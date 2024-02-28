// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

#if CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT == CK_TILE_STATICALLY_INDEXED_ARRAY_USE_TUPLE
namespace detail {
template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<tuple<Xs...>, tuple<Ys...>>
{
    using type = tuple<Xs..., Ys...>;
};

template <typename T, index_t N>
struct statically_indexed_array_impl
{
    using type =
        typename tuple_concat<typename statically_indexed_array_impl<T, N / 2>::type,
                              typename statically_indexed_array_impl<T, N - N / 2>::type>::type;
};

template <typename T>
struct statically_indexed_array_impl<T, 0>
{
    using type = tuple<>;
};

template <typename T>
struct statically_indexed_array_impl<T, 1>
{
    using type = tuple<T>;
};
} // namespace detail

template <typename T, index_t N>
using statically_indexed_array = typename detail::statically_indexed_array_impl<T, N>::type;

#else

// consider mark this struct as deprecated
template <typename T, index_t N>
using statically_indexed_array = array<T, N>;

#endif

// consider always use ck_tile::array for this purpose

template <typename X, typename... Xs>
CK_TILE_HOST_DEVICE constexpr auto make_statically_indexed_array(const X& x, const Xs&... xs)
{
    return statically_indexed_array<X, sizeof...(Xs) + 1>(x, static_cast<X>(xs)...);
}

// make empty statically_indexed_array
template <typename X>
CK_TILE_HOST_DEVICE constexpr auto make_statically_indexed_array()
{
    return statically_indexed_array<X, 0>();
}

} // namespace ck_tile
