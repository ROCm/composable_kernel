// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_NUMBER_HPP
#define CK_NUMBER_HPP

#include "integral_constant.hpp"

namespace ck {

template <index_t N>
using Number = integral_constant<index_t, N>;

template <index_t N>
using LongNumber = integral_constant<long_index_t, N>;

// ---------------------------------------------------------------------------
// is_number<T> -- true if T is a specialization of Number (integral_constant<index_t, N>)
// ---------------------------------------------------------------------------
template <typename T>
struct is_number : false_type
{
};

template <index_t N>
struct is_number<integral_constant<index_t, N>> : true_type
{
};

template <typename T>
inline constexpr bool is_number_v = is_number<T>::value;

// ---------------------------------------------------------------------------
// is_long_number<T> -- true if T is a specialization of LongNumber (integral_constant<long_index_t,
// N>)
// ---------------------------------------------------------------------------
template <typename T>
struct is_long_number : false_type
{
};

template <index_t N>
struct is_long_number<integral_constant<long_index_t, N>> : true_type
{
};

template <typename T>
inline constexpr bool is_long_number_v = is_long_number<T>::value;

} // namespace ck
#endif
