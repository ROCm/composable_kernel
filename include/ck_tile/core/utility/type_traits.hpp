// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include <type_traits>
#include <stdint.h>

namespace ck_tile {

// remove_cvref_t
template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

namespace impl {
template <typename T>
struct is_static_impl
{
    static constexpr bool value = std::is_arithmetic<T>::v ? false : T::is_static();
};
} // namespace impl

template <typename T>
using is_static = impl::is_static_impl<remove_cvref_t<T>>;

template <typename T>
inline constexpr bool is_static_v = is_static<T>::value;

// TODO: deprecate this
template <typename T>
using is_known_at_compile_time = is_static<T>;
// TODO: if evaluating a rvalue, e.g. a const integer
// , this helper will also return false, which is not good(?)
//       do we need something like is_constexpr()?

} // namespace ck_tile
