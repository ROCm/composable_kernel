// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdint>
#include <unordered_set>
#include <numeric>
#include <iterator>

namespace ck {
namespace host {

std::size_t integer_divide_ceil(std::size_t x, std::size_t y);

const std::unordered_set<std::string>& get_xdlop_archs();
// swallow
struct swallow
{
    template <typename... Ts>
    constexpr swallow(Ts&&...)
    {
    }
};

template <typename T>
struct logical_and
{
    constexpr bool operator()(const T& x, const T& y) const { return x && y; }
};

template <typename T>
struct logical_or
{
    constexpr bool operator()(const T& x, const T& y) const { return x || y; }
};

template <typename T>
struct logical_not
{
    constexpr bool operator()(const T& x) const { return !x; }
};
// integral constaqnt
template <class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant type;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

template <typename TX, TX X, typename TY, TY Y>
constexpr auto operator+(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    return integral_constant<decltype(X + Y), X + Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
constexpr auto operator-(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    static_assert(Y <= X, "wrong!");
    return integral_constant<decltype(X - Y), X - Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
constexpr auto operator*(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    return integral_constant<decltype(X * Y), X * Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
constexpr auto operator/(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    static_assert(Y > 0, "wrong!");
    return integral_constant<decltype(X / Y), X / Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
constexpr auto operator%(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    static_assert(Y > 0, "wrong!");
    return integral_constant<decltype(X % Y), X % Y>{};
}
// is _same
template <typename X, typename Y>
struct is_same : public integral_constant<bool, false>
{
};

template <typename X>
struct is_same<X, X> : public integral_constant<bool, true>
{
};

template <typename X, typename Y>
inline constexpr bool is_same_v = is_same<X, Y>::value;
// remove references
template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<std::remove_reference_t<T>>;

template <bool B, typename T = void>
using enable_if = std::enable_if<B, T>;

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T, typename ForwardIterator, typename Size, typename BinaryOperation>
auto accumulate_n(ForwardIterator first, Size count, T init, BinaryOperation op)
    -> decltype(std::accumulate(first, std::next(first, count), init, op))
{
    return std::accumulate(first, std::next(first, count), init, op);
}

// CK Numbers
using long_index_t = int64_t;
using index_t      = int32_t;
template <index_t N>
using Number = integral_constant<index_t, N>;
template <long_index_t N>
using LongNumber = integral_constant<long_index_t, N>;

template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector
{
    using value_t = std::false_type;
    using type    = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...>
{
    using value_t = std::true_type;
    using type    = Op<Args...>;
};

struct nonesuch
{
    ~nonesuch()               = delete;
    nonesuch(nonesuch const&) = delete;
    void operator=(nonesuch const&) = delete;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detector<nonesuch, void, Op, Args...>::value_t;

template <bool isTuple, typename Tensors>
constexpr static auto GetNumABTensors()
{
    if constexpr(isTuple)
    {
        return Number<Tensors::Size()>{};
    }
    else
    {
        return Number<1>{};
    }
}

template <bool predicate, class X, class Y>
struct conditional;

template <class X, class Y>
struct conditional<true, X, Y>
{
    using type = X;
};

template <class X, class Y>
struct conditional<false, X, Y>
{
    using type = Y;
};

template <bool predicate, typename X, typename Y>
constexpr auto conditional_expr(X&& x, Y&& y)
{
    if constexpr(predicate)
    {
        return std::forward<X>(x);
    }
    else
    {
        return std::forward<Y>(y);
    }
}

template <typename T>
struct plus
{
    constexpr T operator()(T a, T b) const { return a + b; }
};

struct multiplies
{
    template <typename A, typename B>
    constexpr auto operator()(const A& a, const B& b) const
    {
        return a * b;
    }
};

template <typename T>
struct equal
{
    constexpr bool operator()(T x, T y) const { return x == y; }
};

template <typename T>
struct less
{
    constexpr bool operator()(T x, T y) const { return x < y; }
};

template <typename T>
struct is_known_at_compile_time;

template <>
struct is_known_at_compile_time<index_t>
{
    static constexpr bool value = false;
};

/**template <typename UpLengths,
          typename Coefficients,
          typename enable_if<UpLengths::Size() == Coefficients::Size(), bool>::type = false>
struct Embed
{
    static constexpr index_t NDimUp = UpLengths::Size();

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    UpLengths up_lengths_;
    Coefficients coefficients_;

    constexpr Embed() = default;

    constexpr Embed(const UpLengths& up_lengths, const Coefficients& coefficients)
        : up_lengths_{up_lengths}, coefficients_{coefficients}
    {
    }

    static constexpr index_t GetNumOfLowerDimension() { return 1; }

    static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    constexpr void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}([&idx_low, &idx_up, this](auto i) {
            idx_low(Number<0>{}) += idx_up[i] * this->coefficients_[i];
        });
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                          const UpIdxDiff& idx_diff_up,
                          LowIdx& idx_low,
                          const UpIdx&,
                          Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == NDimUp &&
                          LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}(
            [&](auto i) { idx_diff_low(Number<0>{}) += idx_diff_up[i] * coefficients_[i]; });

        idx_low += idx_diff_low;
    }

    static constexpr bool IsLinearTransform() { return true; }

    static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex() { return true; }

    template <typename UpIdx>
    static constexpr bool IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& )//idx_up
    {
        return true;
    }

    static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<Coefficients>::value;
    }
};
template <typename UpLengths,
          typename Coefficients,
          typename enable_if<UpLengths::Size() == Coefficients::Size(), bool>::type = false>
constexpr auto make_embed_transform(const UpLengths& up_lengths, const Coefficients& coefficients)
{
    return Embed<UpLengths, Coefficients>{up_lengths, coefficients};
}**/
} // namespace host
} // namespace ck
