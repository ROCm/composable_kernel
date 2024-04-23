// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include "ck_tile/core/utility/iterator.hpp"

namespace ck_tile {
namespace ranges {
template <typename R>
using iterator_t = decltype(std::begin(std::declval<R&>()));

template <typename R>
using sentinel_t = decltype(std::end(std::declval<R&>()));

template <typename R>
using range_size_t = decltype(std::size(std::declval<R&>()));

template <typename R>
using range_difference_t = ck_tile::iter_difference_t<ranges::iterator_t<R>>;

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

template <typename Range>
inline constexpr bool is_sized_range_v = is_sized_range<Range>::value;

template <typename Range, typename = void>
struct is_bidirectional_range : std::false_type
{
};

template <typename Range>
struct is_bidirectional_range<Range, std::void_t<>>
    : std::bool_constant<
          is_range_v<Range> &&
          is_bidirectional_iterator_v<remove_cvref_t<decltype(begin(std::declval<Range>()))>>>
{
};

template <typename Range>
inline constexpr bool is_bidirectional_range_v = is_bidirectional_range<Range>::value;

template <typename Range, typename = void>
struct is_random_access_range : std::false_type
{
};

template <typename Range>
struct is_random_access_range<Range, std::void_t<>>
    : std::bool_constant<
          is_range_v<Range> &&
          is_random_access_iterator_v<remove_cvref_t<decltype(begin(std::declval<Range>()))>>>
{
};

template <typename Range>
inline constexpr bool is_random_access_range_v = is_random_access_range<Range>::value;

} // namespace ranges
} // namespace ck_tile
