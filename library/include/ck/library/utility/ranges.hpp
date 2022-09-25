// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include "ck/library/utility/concepts.hpp"
#include "ck/library/utility/iterator.hpp"

namespace ck {

struct from_range_t
{
    explicit from_range_t() = default;
};

inline constexpr from_range_t from_range{};

namespace ranges {

template <typename R>
using iterator_t = decltype(std::begin(std::declval<R&>()));

template <typename R>
using sentinel_t = decltype(std::end(std::declval<R&>()));

template <typename R>
using range_size_t = decltype(std::size(std::declval<R&>()));

template <typename R>
using range_difference_t = ck::iter_difference_t<ranges::iterator_t<R>>;

template <typename R>
using range_value_t = iter_value_t<ranges::iterator_t<R>>;

template <typename R>
using range_reference_t = iter_reference_t<ranges::iterator_t<R>>;

template <typename T, typename = void>
struct is_range : std::false_type
{
};

template <typename T>
struct is_range<
    T,
    std::void_t<decltype(std::begin(std::declval<T&>())), decltype(std::end(std::declval<T&>()))>>
    : std::true_type
{
};

template <typename T>
inline constexpr bool is_range_v = is_range<T>::value;

template <typename T, typename = void>
struct is_sized_range : std::false_type
{
};

template <typename T>
struct is_sized_range<T, std::void_t<decltype(std::size(std::declval<T&>()))>>
    : std::bool_constant<is_range_v<T>>
{
};

template <typename T>
inline constexpr bool is_sized_range_v = is_sized_range<T>::value;

template <typename T>
struct is_common_range
    : std::bool_constant<is_range_v<T> && std::is_same_v<iterator_t<T>, sentinel_t<T>>>
{
};

template <typename T>
inline constexpr bool is_common_range_v = is_common_range<T>::value;

template <typename Cont, typename Range, typename... Args>
auto to(Range&& range, Args&&... args)
    -> std::enable_if_t<is_convertible_to_v<range_reference_t<Range>, range_value_t<Range>> &&
                            is_constructible_from_v<Cont, Range, Args...>,
                        Cont>
{
    return Cont(std::forward<Range>(range), std::forward<Args>(args)...);
}

template <typename Cont, typename Range, typename... Args>
auto to(Range&& range, Args&&... args)
    -> std::enable_if_t<is_convertible_to_v<range_reference_t<Range>, range_value_t<Range>> &&
                            !is_constructible_from_v<Cont, Range, Args...> &&
                            is_constructible_from_v<Cont, from_range_t, Range, Args...>,
                        Cont>
{
    return Cont(from_range, std::forward<Range>(range), std::forward<Args>(args)...);
}

template <typename Cont, typename Range, typename... Args>
auto to(Range&& range, Args&&... args) -> std::enable_if_t<
    is_convertible_to_v<range_reference_t<Range>, range_value_t<Range>> &&
        !is_constructible_from_v<Cont, Range, Args...> &&
        !is_constructible_from_v<Cont, from_range_t, Range, Args...> &&
        (is_common_range_v<Range> &&
         is_constructible_from_v<Cont, iterator_t<Range>, sentinel_t<Range>, Args...>),
    Cont>
{
    return Cont(std::begin(std::forward<Range>(range)),
                std::end(std::forward<Range>(range)),
                std::forward<Args>(args)...);
}
} // namespace ranges
} // namespace ck
