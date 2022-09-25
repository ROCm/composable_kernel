// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>

namespace ck {

template <typename T>
struct is_destructible : std::is_nothrow_destructible<T>
{
};

template <typename T>
inline constexpr bool is_destructible_v = is_destructible<T>::value;

template <typename T, typename... Args>
struct is_constructible_from
    : std::bool_constant<is_destructible_v<T> && std::is_constructible_v<T, Args...>>
{
};

template <typename T, typename... Args>
inline constexpr bool is_constructible_from_v = is_constructible_from<T, Args...>::value;

template <typename From, typename To, typename = void>
struct is_convertible_to : std::false_type
{
};

template <typename From, typename To>
struct is_convertible_to<From, To, std::void_t<decltype(static_cast<To>(std::declval<From>()))>>
    : std::bool_constant<std::is_convertible_v<From, To>>
{
};

template <typename From, typename To>
inline constexpr bool is_convertible_to_v = is_convertible_to<From, To>::value;

} // namespace ck
