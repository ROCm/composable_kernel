// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_ARRAY_MULTI_INDEX_HPP
#define CK_ARRAY_MULTI_INDEX_HPP

#include "common_header.hpp"

namespace ck {

template <index_t N, typename T = index_t>
using MultiIndex = Array<T, N>;

template <typename IdxType = index_t, typename... Xs>
__host__ __device__ constexpr auto make_multi_index(Xs&&... xs)
{
    return make_array<IdxType>(IdxType{xs}...);
}

template <index_t NSize, typename IdxType = index_t>
__host__ __device__ constexpr auto make_zero_multi_index()
{
    return unpack([](auto... xs) { return make_multi_index<IdxType>(xs...); },
                  typename uniform_sequence_gen<NSize, 0>::type{});
}

template <typename T>
__host__ __device__ constexpr auto to_multi_index(const T& x)
{
    return unpack([](auto... ys) { return make_multi_index(ys...); }, x);
}

template <index_t NSize, typename X>
__host__ __device__ constexpr auto operator+=(MultiIndex<NSize>& y, const X& x)
{
    static_assert(X::Size() == NSize, "wrong! size not the same");
    static_for<0, NSize, 1>{}([&](auto i) { y(i) += x[i]; });
    return y;
}

template <index_t NSize, typename X>
__host__ __device__ constexpr auto operator-=(MultiIndex<NSize>& y, const X& x)
{
    static_assert(X::Size() == NSize, "wrong! size not the same");
    static_for<0, NSize, 1>{}([&](auto i) { y(i) -= x[i]; });
    return y;
}

template <index_t NSize, typename T>
__host__ __device__ constexpr auto operator+(const MultiIndex<NSize>& a, const T& b)
{
    using type = MultiIndex<NSize>;
    static_assert(T::Size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a[i] + b[i]; });
    return r;
}

template <index_t NSize, typename T>
__host__ __device__ constexpr auto operator-(const MultiIndex<NSize>& a, const T& b)
{
    using type = MultiIndex<NSize>;
    static_assert(T::Size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a[i] - b[i]; });
    return r;
}

template <index_t NSize, typename T>
__host__ __device__ constexpr auto operator*(const MultiIndex<NSize>& a, const T& b)
{
    using type = MultiIndex<NSize>;
    static_assert(T::Size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a[i] * b[i]; });
    return r;
}

} // namespace ck
#endif
