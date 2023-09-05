// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/remove_cvref.hpp"

namespace ck {

namespace detail {

template <typename T>
struct is_static_impl
{
    static constexpr bool value = T::IsStatic();
};

template <>
struct is_static_impl<int32_t>
{
    static constexpr bool value = false;
};

template <>
struct is_static_impl<int64_t>
{
    static constexpr bool value = false;
};

} // namespace detail

template <typename T>
using is_static = detail::is_static_impl<remove_cvref_t<T>>;

template <typename T>
inline constexpr bool is_static_v = is_static<T>::value;

// TODO: deprecate this
template <typename T>
using is_known_at_compile_time = is_static<T>;

} // namespace ck
