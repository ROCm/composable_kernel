// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

#include <type_traits>

namespace ck_tile {

// this structure is used to pick up the <base> type inside
// using xxx = <base> __attribute__((ext_vector_type(N)));
// because clang only allow native type + bool in this term (custom type will fail)
// overload this structure to let proper <base> type

template <typename T>
struct native_t
{
    using type = remove_cvref_t<T>;
};

// we name this as ext_vector purposely, because clang ext_vector_type extention only accept literay
// basic type to construct a ext_vector_type you must be very careful using this, or will have lot
// of compiler errors e.g. struct A; using Ax2_t = A __attribute__((ext_vector_type(2)));  -> will
// have compiler error
namespace impl {

template <typename T_, index_t N_, typename = void>
struct ext_vector;

template <typename T_, index_t N_>
struct ext_vector<T_, N_, std::enable_if_t<!std::is_class_v<typename native_t<T_>::type>>>
{
    static constexpr index_t N = N_;
    // struct type is not supported for ext_vector
    using value_type = typename native_t<T_>::type;
    static_assert(!std::is_class_v<value_type>);
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};

template <typename T_, index_t N_>
struct ext_vector<T_, N_, std::enable_if_t<std::is_class_v<typename native_t<T_>::type>>>
{
    static constexpr index_t N = N_;
    // struct type is not supported for ext_vector
    using value_type = typename native_t<T_>::type::type;
    static_assert(!std::is_class_v<value_type>);
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};

template <typename V_, index_t Vs_, index_t N_>
struct ext_vector<V_ __attribute__((ext_vector_type(Vs_))),
                  N_,
                  std::enable_if_t<!std::is_class_v<typename native_t<V_>::type>>>
{
    static constexpr index_t N = Vs_ * N_;
    using value_type           = typename native_t<remove_cvref_t<V_>>::type;
    static_assert(!std::is_class_v<value_type>);
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};

template <typename V_, index_t Vs_, index_t N_>
struct ext_vector<V_ __attribute__((ext_vector_type(Vs_))),
                  N_,
                  std::enable_if_t<std::is_class_v<typename native_t<V_>::type>>>
{
    static constexpr index_t N = Vs_ * N_;
    using value_type           = typename native_t<remove_cvref_t<V_>>::type::type;
    static_assert(!std::is_class_v<value_type>);
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};

} // namespace impl

template <typename T, index_t N>
using ext_vector_t = typename impl::ext_vector<T, N>::type;

} // namespace ck_tile
