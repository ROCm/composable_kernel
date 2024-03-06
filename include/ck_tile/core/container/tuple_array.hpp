// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

namespace detail {
template <typename T, index_t N>
struct tuple_array_impl
{
    using type = typename tuple_concat<typename tuple_array_impl<T, N / 2>::type,
                                       typename tuple_array_impl<T, N - N / 2>::type>::type;
};

template <typename T>
struct tuple_array_impl<T, 0>
{
    using type = tuple<>;
};

template <typename T>
struct tuple_array_impl<T, 1>
{
    using type = tuple<T>;
};
} // namespace detail

template <typename T, index_t N>
using tuple_array_base_t = typename detail::tuple_array_impl<T, N>::type;

template <typename T_, index_t N_>
struct tuple_array : tuple_array_base_t<T_, N_>
{
    using value_type           = T_;
    static constexpr index_t N = N_;

// clang-format off
#define TA_COM_() static_assert(sizeof(value_type) * N % sizeof(Tx) == 0); constexpr int vx = sizeof(value_type) * N / sizeof(Tx)
    template <typename Tx> CK_TILE_HOST_DEVICE constexpr decltype(auto) get_as()                            { TA_COM_(); return reinterpret_cast<tuple_array<Tx, vx>&>(*this); }
    template <typename Tx> CK_TILE_HOST_DEVICE constexpr decltype(auto) get_as() const                      { TA_COM_(); return reinterpret_cast<const tuple_array<Tx, vx>&>(*this); }

    // below index is for index *AFTER* type convert, not before
    template <typename Tx> CK_TILE_HOST_DEVICE constexpr decltype(auto) get_as(index_t i)                   { TA_COM_(); return reinterpret_cast<tuple_array<Tx, vx>&>(*this).at(i); }
    template <typename Tx> CK_TILE_HOST_DEVICE constexpr decltype(auto) get_as(index_t i) const             { TA_COM_(); return reinterpret_cast<const tuple_array<Tx, vx>&>(*this).at(i); }
    template <typename Tx, index_t I> CK_TILE_HOST_DEVICE constexpr decltype(auto) get_as(number<I>)        { TA_COM_(); return reinterpret_cast<tuple_array<Tx, vx>&>(*this).at(number<I>{}); }
    template <typename Tx, index_t I> CK_TILE_HOST_DEVICE constexpr decltype(auto) get_as(number<I>) const  { TA_COM_(); return reinterpret_cast<const tuple_array<Tx, vx>&>(*this).at(number<I>{}); }
    
    template <typename Tx> CK_TILE_HOST_DEVICE constexpr void set_as(index_t i, const Tx & x)               { TA_COM_(); reinterpret_cast<tuple_array<Tx, vx>&>(*this).at(i) = x; }
    template <typename Tx, index_t I> CK_TILE_HOST_DEVICE constexpr void set_as(number<I>, const Tx & x)    { TA_COM_(); reinterpret_cast<tuple_array<Tx, vx>&>(*this).at(number<I>{}) = x; }
#undef TA_COM_
    // clang-format on
};

} // namespace ck_tile
