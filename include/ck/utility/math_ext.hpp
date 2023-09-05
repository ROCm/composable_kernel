// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

// Here should use MultiIndex<NSize>, instead of Tuple<Ys...>, although the former
// is the alias of the latter. This is because compiler cannot infer the NSize if
// using MultiIndex<NSize>
// TODO: how to fix this?
template <
    typename... Ys,
    typename X,
    enable_if_t<!std::is_integral<X>::value && !std::is_floating_point<X>::value, bool> = false>
__host__ __device__ constexpr auto operator+=(Tuple<Ys...>& y, const X& x)
{
    static_assert(X::Size() == sizeof...(Ys), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Ys);
    static_for<0, NSize, 1>{}([&](auto i) { y(i) += x[i]; });
    return y;
}

template <
    typename... Ys,
    typename X,
    enable_if_t<!std::is_integral<X>::value && !std::is_floating_point<X>::value, bool> = false>
__host__ __device__ constexpr auto operator-=(Tuple<Ys...>& y, const X& x)
{
    static_assert(X::Size() == sizeof...(Ys), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Ys);
    static_for<0, NSize, 1>{}([&](auto i) { y(i) -= x[i]; });
    return y;
}

template <
    typename... Xs,
    typename Y,
    enable_if_t<!std::is_integral<Y>::value && !std::is_floating_point<Y>::value, bool> = false>
__host__ __device__ constexpr auto operator+(const Tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = x[i] + y[i]; });
    return r;
}

template <
    typename... Xs,
    typename Y,
    enable_if_t<!std::is_integral<Y>::value && !std::is_floating_point<Y>::value, bool> = false>
__host__ __device__ constexpr auto operator-(const Tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = x[i] - y[i]; });
    return r;
}

template <
    typename... Xs,
    typename Y,
    enable_if_t<!std::is_integral<Y>::value && !std::is_floating_point<Y>::value, bool> = false>
__host__ __device__ constexpr auto operator*(const Tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = x[i] * y[i]; });
    return r;
}

// MultiIndex = scalar * MultiIndex
template <typename... Xs,
          typename Y,
          enable_if_t<std::is_integral<Y>::value || std::is_floating_point<Y>::value, bool> = false>
__host__ __device__ constexpr auto operator*(Y a, const Tuple<Xs...>& x)
{
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a * x[i]; });
    return r;
}

// MultiIndex = MultiIndex * scalar
template <typename... Xs,
          typename Y,
          enable_if_t<std::is_integral<Y>::value || std::is_floating_point<Y>::value, bool> = false>
__host__ __device__ constexpr auto operator*(const Tuple<Xs...>& x, Y a)
{
    return a * x;
}

template <typename... Xs, typename... Ys>
__host__ __device__ constexpr auto operator/(const Tuple<Xs...>& x, const Tuple<Ys...>& y)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong!");

    constexpr index_t NSize = sizeof...(Xs);

    return generate_tuple([&](auto i) { return x[i] / y[i]; }, Number<NSize>{});
}

namespace mathext {

template <typename... Xs>
__host__ __device__ constexpr auto exp(const Tuple<Xs...>& x)
{
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = math::exp(x[i]); });
    return r;
}

template <typename... Xs, typename Y>
__host__ __device__ constexpr auto max(const Tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = math::max(x[i], y[i]); });
    return r;
}

} // namespace mathext
} // namespace ck
