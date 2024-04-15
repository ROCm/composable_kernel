// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

namespace ck_tile {

template <typename T>
using iter_value_t = typename std::iterator_traits<remove_cvref_t<T>>::value_type;

template <typename T>
using iter_reference_t = decltype(*std::declval<T&>());

template <typename T>
using iter_difference_t = typename std::iterator_traits<remove_cvref_t<T>>::difference_type;

template <typename T, typename = void>
struct is_iterator : std::false_type
{
};

template <typename T>
struct is_iterator<T,
                   std::void_t<decltype(*std::declval<T>()),
                               decltype(++std::declval<std::add_lvalue_reference_t<T>>()),
                               decltype(std::declval<std::add_lvalue_reference_t<T>>()++)>>
    : std::true_type
{
};

template <typename T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;

namespace detail {
struct Placeholder final
{
    template <typename T>
    constexpr inline operator T() const noexcept;
};
} // namespace detail

template <typename Iterator, typename = void>
struct is_output_iterator : std::false_type
{
};

template <typename Iterator>
struct is_output_iterator<
    Iterator,
    std::void_t<decltype(*std::declval<Iterator>() = std::declval<detail::Placeholder>())>>
    : std::bool_constant<is_iterator_v<Iterator>>
{
};

template <typename T>
inline constexpr bool is_output_iterator_v = is_output_iterator<T>::value;

template <typename Iterator, typename = void>
struct is_bidirectional_iterator : std::false_type
{
};

template <typename Iterator>
struct is_bidirectional_iterator<
    Iterator,
    std::void_t<decltype(--std::declval<std::add_lvalue_reference_t<Iterator>>()),
                decltype(std::declval<std::add_lvalue_reference_t<Iterator>>()--)>>
    : std::bool_constant<is_iterator_v<Iterator>>
{
};

template <typename Iterator>
inline constexpr bool is_bidirectional_iterator_v = is_bidirectional_iterator<Iterator>::value;

template <typename Iterator, typename = void>
struct is_random_access_iterator : std::false_type
{
};

template <typename Iterator>
struct is_random_access_iterator<Iterator,
                                 std::void_t<decltype(std::declval<Iterator>() + 1),
                                             decltype(std::declval<Iterator>() - 1),
                                             decltype(std::declval<Iterator>()[1])>>
    : std::bool_constant<is_iterator_v<Iterator>>
{
};

template <typename Iterator>
inline constexpr bool is_random_access_iterator_v = is_random_access_iterator<Iterator>::value;

} // namespace ck_tile
