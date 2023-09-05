// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/integral_constant.hpp"
#include "ck/utility/enable_if.hpp"

namespace ck {

// is_same
static_assert(__has_builtin(__is_same), "");

template <typename X, typename Y>
using is_same = integral_constant<bool, __is_same(X, Y)>;

template <typename X, typename Y>
inline constexpr bool is_same_v = is_same<X, Y>::value;

static_assert(__has_builtin(__type_pack_element), "");

// type_pack_element
template <index_t I, typename... Ts>
using type_pack_element = __type_pack_element<I, Ts...>;

// is_pointer
template <typename T>
inline constexpr bool is_pointer_v = std::is_pointer<T>::value;

// is_empty
template <typename T>
inline constexpr bool is_empty_v = std::is_empty<T>::value;

// bit_cast
template <typename Y, typename X, typename enable_if<sizeof(X) == sizeof(Y), bool>::type = false>
__host__ __device__ constexpr Y bit_cast(const X& x)
{
    static_assert(__has_builtin(__builtin_bit_cast), "");

    return __builtin_bit_cast(Y, x);
}

} // namespace ck
