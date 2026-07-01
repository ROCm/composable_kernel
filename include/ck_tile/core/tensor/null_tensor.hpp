// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

struct null_tensor
{
};

// utility to check if this is a Null Tensor
namespace impl {
template <typename>
struct is_null_tensor : public std::false_type
{
};

template <>
struct is_null_tensor<null_tensor> : public std::true_type
{
};
} // namespace impl

template <typename T>
constexpr bool is_null_tensor_v = impl::is_null_tensor<remove_cvref_t<T>>::value;

} // namespace ck_tile
