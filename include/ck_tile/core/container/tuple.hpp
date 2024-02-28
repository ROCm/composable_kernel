// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include <utility>

namespace ck_tile {

namespace impl {

template <index_t idx, typename T, bool is_empty = std::is_empty_v<T>>
struct tuple_element
{
};

template <index_t idx, typename T>
struct tuple_element<idx, T, true>
{
    CK_TILE_HOST_DEVICE constexpr tuple_element() {}
    CK_TILE_HOST_DEVICE constexpr tuple_element(const T&) {}
};

template <index_t idx, typename T>
struct tuple_element<idx, T, false>
{
    CK_TILE_HOST_DEVICE constexpr tuple_element() {}
    CK_TILE_HOST_DEVICE constexpr tuple_element(const T& e) : element(e) {}
    T element;
};

template <std::size_t I, class T>
CK_TILE_HOST_DEVICE constexpr T const& getv(tuple_element<I, T, false> const& x)
{
    return x.element;
}

template <std::size_t I, class T>
CK_TILE_HOST_DEVICE constexpr T& getv(tuple_element<I, T, false>& x)
{
    return x.element;
}

template <std::size_t I, class T>
CK_TILE_HOST_DEVICE constexpr T&& getv(tuple_element<I, T, false>&& x)
{
    return static_cast<T&&>(x.element);
}

template <typename index_seq, typename... T>
struct tuple_base;

template <index_t... I, typename... T>
struct tuple_base<sequence<I...>, T...> : public tuple_element<I, T>...
{
    CK_TILE_HOST_DEVICE constexpr tuple_base() {}

    template <class... U>
    CK_TILE_HOST_DEVICE constexpr explicit tuple_base(U const&... u) : tuple_element<I, T>(u)...
    {
    }

    template <class... U>
    CK_TILE_HOST_DEVICE constexpr tuple_base(tuple_base<sequence<I...>, U...> const& u)
        : tuple_element<I, T>(getv(static_cast<tuple_element<I, U> const&>(u)))...
    {
    }
};
} // namespace impl

template <class... T>
struct tuple : impl::tuple_base<make_index_sequence<sizeof...(T)>, T...>
{
    CK_TILE_HOST_DEVICE
    static constexpr auto size() { return sizeof...(T); }
    using base = impl::tuple_base<make_index_sequence<sizeof...(T)>, T...>;
    CK_TILE_HOST_DEVICE constexpr tuple() {}

    template <class... U>
    CK_TILE_HOST_DEVICE constexpr tuple(U const&... u)
        : impl::tuple_base<make_index_sequence<sizeof...(T)>, T...>(u...)
    {
    }

    template <class... U>
    CK_TILE_HOST_DEVICE constexpr tuple(tuple<U...> const& u)
        : impl::tuple_base<make_index_sequence<sizeof...(T)>, T...>(
              static_cast<impl::tuple_base<U...> const&>(u))
    {
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_static()
    {
        bool flag = true;

        static_for<0, sizeof...(Xs), 1>{}([&flag](auto i) {
            flag &= is_static_v<remove_cvref_t<__type_pack_element<i.value, Xs...>>>;
        });

        return flag;
    }

#define TP_COM_() static_assert(I < size(), "wrong! out of range")
    // clang-format off
    template<index_t I> CK_TILE_HOST_DEVICE constexpr const auto & get() const          { TP_COM_(); return impl::getv<I>(*this); }
    template<index_t I> CK_TILE_HOST_DEVICE constexpr const auto & get(number<I>) const { TP_COM_(); return get<I>(); }
    template<index_t I> CK_TILE_HOST_DEVICE constexpr auto & get()                      { TP_COM_(); return impl::getv<I>(*this); }
    template<index_t I> CK_TILE_HOST_DEVICE constexpr auto & get(number<I>)             { TP_COM_(); return get<I>(); }

    template<index_t I> CK_TILE_HOST_DEVICE constexpr const auto & at() const          { TP_COM_(); return impl::getv<I>(*this); }
    template<index_t I> CK_TILE_HOST_DEVICE constexpr const auto & at(number<I>) const { TP_COM_(); return get<I>(); }
    template<index_t I> CK_TILE_HOST_DEVICE constexpr auto & at()                      { TP_COM_(); return impl::getv<I>(*this); }
    template<index_t I> CK_TILE_HOST_DEVICE constexpr auto & at(number<I>)             { TP_COM_(); return get<I>(); }

    template<index_t I> CK_TILE_HOST_DEVICE constexpr auto & operator[](number<I>)             { TP_COM_(); return get<I>(); }
    template<index_t I> CK_TILE_HOST_DEVICE constexpr const auto & operator[](number<I>) const { TP_COM_(); return get<I>(); }
    template<index_t I> CK_TILE_HOST_DEVICE constexpr auto & operator()(number<I>)             { TP_COM_(); return get<I>(); }  // TODO: compatible
    // clang-format on
#undef TP_COM_
};

// template <class... T>
// CK_TILE_HOST_DEVICE constexpr
// tuple<T...>
// make_tuple(T const&... t)
// {
//     return {t...};
// }

template <typename... Xs>
CK_TILE_HOST_DEVICE constexpr auto make_tuple(Xs&&... xs)
{
    return tuple<remove_cvref_t<Xs>...>(std::forward<Xs>(xs)...);
}

// https://en.cppreference.com/w/cpp/utility/tuple/tie
template <typename... Args>
constexpr tuple<Args&...> tie(Args&... args) noexcept
{
    return {args...};
}

template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<tuple<Xs...>, tuple<Ys...>>
{
    using type = tuple<Xs..., Ys...>;
};

template <typename F, index_t N>
CK_TILE_HOST_DEVICE constexpr auto generate_tuple(F&& f, number<N>)
{
    return unpack([&f](auto&&... is) { return make_tuple(f(is)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <typename F, index_t N>
CK_TILE_HOST_DEVICE constexpr auto generate_tie(F&& f, number<N>)
{
    return unpack([&f](auto&&... is) { return tie(f(is)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

// tx and ty are tuple of references, return type of will tuple of referennce (not rvalue)
template <typename... X, typename... Y>
CK_TILE_HOST_DEVICE constexpr auto concat_tuple_of_reference(const tuple<X&...>& tx,
                                                             const tuple<Y&...>& ty)
{
    return unpack2(
        [&](auto&&... zs) { return tuple<decltype(zs)...>{std::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

template <typename... X, typename... Y>
CK_TILE_HOST_DEVICE constexpr auto concat_tuple(const tuple<X...>& tx, const tuple<Y...>& ty)
{
    return unpack2(
        [&](auto... zs) { return tuple<decltype(zs)...>{std::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

// Support any number of tuples to concat (also 1)
template <typename... X>
CK_TILE_HOST_DEVICE constexpr auto concat_tuple(const tuple<X...>& tx)
{
    return tx;
}

template <typename... X, typename... Tuples>
CK_TILE_HOST_DEVICE constexpr auto concat_tuple(const tuple<X...>& tx, const Tuples&... tuples)
{
    return concat_tuple(tx, concat_tuple(tuples...));
}

namespace detail {

template <typename F, typename X, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto transform_tuples_impl(F f, const X& x, sequence<Is...>)
{
    return make_tuple(f(x.at(number<Is>{}))...);
}

template <typename F, typename X, typename Y, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, sequence<Is...>)
{
    return make_tuple(f(x.at(number<Is>{}), y.at(number<Is>{}))...);
}

template <typename F, typename X, typename Y, typename Z, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, const Z& z, sequence<Is...>)
{
    return make_tuple(f(x.at(number<Is>{}), y.at(number<Is>{}), z.at(number<Is>{}))...);
}

} // namespace detail

template <typename F, typename X>
CK_TILE_HOST_DEVICE constexpr auto transform_tuples(F f, const X& x)
{
    return detail::transform_tuples_impl(
        f, x, typename arithmetic_sequence_gen<0, X::size()(), 1>::type{});
}

template <typename F, typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto transform_tuples(F f, const X& x, const Y& y)
{
    return detail::transform_tuples_impl(
        f, x, y, typename arithmetic_sequence_gen<0, X::size()(), 1>::type{});
}

template <typename F, typename X, typename Y, typename Z>
CK_TILE_HOST_DEVICE constexpr auto transform_tuples(F f, const X& x, const Y& y, const Z& z)
{
    return detail::transform_tuples_impl(
        f, x, y, z, typename arithmetic_sequence_gen<0, X::size()(), 1>::type{});
}

// By default unroll to the flatten
template <index_t Depth = 0, index_t MaxDepth = -1>
CK_TILE_HOST_DEVICE constexpr auto unroll_nested_tuple(const tuple<>& element)
{
    return element;
}

template <index_t Depth = 0, index_t MaxDepth = -1, typename T>
CK_TILE_HOST_DEVICE constexpr auto unroll_nested_tuple(const T& element)
{
    return make_tuple(element);
}

template <index_t Depth = 0, index_t MaxDepth = -1, typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto unroll_nested_tuple(const tuple<Ts...>& tuple)
{
    if constexpr(Depth == MaxDepth)
    {
        return tuple;
    }
    else
    {
        return unpack(
            [&](auto&&... ts) {
                return concat_tuple(unroll_nested_tuple<Depth + 1, MaxDepth>(ts)...);
            },
            tuple);
    }
}

template <typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto tuple_reverse(const tuple<Ts...>& tuple)
{
    return generate_tuple(
        [&](auto i) {
            using Idx = number<tuple<Ts...>::size()() - i - 1>;
            return tuple.at(Idx{});
        },
        number<tuple<Ts...>::size()()>{});
}

// Reduce tuple values in specific range using Function
template <index_t Idx, index_t End, typename F, typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto tuple_reduce(F&& f, const tuple<Ts...>& tuple)
{
    static_assert(Idx < End, "Wrong parameters for tuple_reduce");
    if constexpr(Idx + 1 == End)
    {
        return tuple.at(number<Idx>{});
    }
    else
    {
        return f(tuple.at(number<Idx>{}), tuple_reduce<Idx + 1, End>(f, tuple));
    }
}

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());

template <typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto is_nested_tuple(const tuple<Ts...>&)
{
    return (is_detected<is_tuple, Ts>::value || ...);
}

template <index_t depth = 0, typename T>
CK_TILE_HOST_DEVICE constexpr auto tuple_depth(const T&)
{
    return depth;
}

template <index_t depth = 0, typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto tuple_depth(const tuple<Ts...>&)
{
    return math::max(tuple_depth<depth + 1>(Ts{})...);
}

template <typename... Seqs>
CK_TILE_HOST_DEVICE constexpr auto to_array_of_array(tuple<Seqs...> t_of_s)
{
    constexpr index_t n0 = sizeof...(Seqs);

    constexpr index_t max_n1 = [&] {
        index_t max_n1_ = 0;

        static_for<0, n0, 1>{}([&](auto i0) {
            constexpr index_t n1 = t_of_s[i0].size()();

            max_n1_ = max_n1_ < n1 ? n1 : max_n1_;
        });

        return max_n1_;
    }();

    array<array<index_t, max_n1>, n0> a_of_a{{-1}};

    static_for<0, n0, 1>{}([&](auto i0) {
        constexpr index_t n1 = t_of_s[i0].size()();

        static_for<0, n1, 1>{}([&](auto i1) { a_of_a(i0)(i1) = t_of_s[i0][i1]; });
    });

    return a_of_a;
}

// Here should use MultiIndex<NSize>, instead of tuple<Ys...>, although the former
// is the alias of the latter. This is because compiler cannot infer the NSize if
// using MultiIndex<NSize>
// TODO: how to fix this?
template <typename... Ys,
          typename X,
          std::enable_if_t<!std::is_integral<X>::value && !std::is_floating_point<X>::value, bool> =
              false>
CK_TILE_HOST_DEVICE constexpr auto operator+=(tuple<Ys...>& y, const X& x)
{
    static_assert(X::Size() == sizeof...(Ys), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Ys);
    static_for<0, NSize, 1>{}([&](auto i) { y[i] += x[i]; });
    return y;
}

template <typename... Ys,
          typename X,
          std::enable_if_t<!std::is_integral<X>::value && !std::is_floating_point<X>::value, bool> =
              false>
CK_TILE_HOST_DEVICE constexpr auto operator-=(tuple<Ys...>& y, const X& x)
{
    static_assert(X::Size() == sizeof...(Ys), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Ys);
    static_for<0, NSize, 1>{}([&](auto i) { y[i] -= x[i]; });
    return y;
}

template <typename... Xs,
          typename Y,
          std::enable_if_t<!std::is_integral<Y>::value && !std::is_floating_point<Y>::value, bool> =
              false>
CK_TILE_HOST_DEVICE constexpr auto operator+(const tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = x[i] + y[i]; });
    return r;
}

template <typename... Xs,
          typename Y,
          std::enable_if_t<!std::is_integral<Y>::value && !std::is_floating_point<Y>::value, bool> =
              false>
CK_TILE_HOST_DEVICE constexpr auto operator-(const tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = x[i] - y[i]; });
    return r;
}

template <typename... Xs,
          typename Y,
          std::enable_if_t<!std::is_integral<Y>::value && !std::is_floating_point<Y>::value, bool> =
              false>
CK_TILE_HOST_DEVICE constexpr auto operator*(const tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = x[i] * y[i]; });
    return r;
}

// MultiIndex = scalar * MultiIndex
template <
    typename... Xs,
    typename Y,
    std::enable_if_t<std::is_integral<Y>::value || std::is_floating_point<Y>::value, bool> = false>
CK_TILE_HOST_DEVICE constexpr auto operator*(Y a, const tuple<Xs...>& x)
{
    constexpr index_t NSize = sizeof...(Xs);
    tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = a * x[i]; });
    return r;
}

// MultiIndex = MultiIndex * scalar
template <
    typename... Xs,
    typename Y,
    std::enable_if_t<std::is_integral<Y>::value || std::is_floating_point<Y>::value, bool> = false>
CK_TILE_HOST_DEVICE constexpr auto operator*(const tuple<Xs...>& x, Y a)
{
    return a * x;
}

template <typename... Xs, typename... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator/(const tuple<Xs...>& x, const tuple<Ys...>& y)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong!");
    constexpr index_t NSize = sizeof...(Xs);
    return generate_tuple([&](auto i) { return x[i] / y[i]; }, number<NSize>{});
}

} // namespace ck_tile

// WARNING: needed by compiler for C++ structured binding support only, don't use this
namespace std {

template <typename... Ts>
struct tuple_size<ck_tile::tuple<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)>
{
};

template <std::size_t I, typename... Ts>
struct tuple_element<I, ck_tile::tuple<Ts...>> : ck_tile::tuple_element<I, ck_tile::tuple<Ts...>>
{
};

template <typename... Ts>
struct tuple_size<const ck_tile::tuple<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)>
{
};

template <std::size_t I, typename... Ts>
struct tuple_element<I, const ck_tile::tuple<Ts...>>
    : ck_tile::tuple_element<I, const ck_tile::tuple<Ts...>>
{
};

} // namespace std
