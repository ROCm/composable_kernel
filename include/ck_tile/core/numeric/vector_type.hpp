// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"

namespace ck_tile {

// TODO: the whole content of this file should consider deprecated!
template <typename T_, index_t N_>
struct vector_type
{
    static constexpr index_t N = N_;
    using value_type           = T_;
    using type                 = value_type __attribute__((ext_vector_type(N))); // this is danguous

    CK_HOST_DEVICE constexpr vector_type()
    {
        for(auto i = 0; i < N; i++)
            data[i] = static_cast<value_type>(0);
    }
    CK_HOST_DEVICE constexpr vector_type(type v)
    {
        auto& r = reinterpret_cast<const array<value_type, N>&>(v);
        for(auto i = 0; i < N; i++)
            data[i] = r.get(i);
    }

    value_type data[N];
    CK_HOST_DEVICE static constexpr auto size() { return N; }
    CK_HOST_DEVICE auto& get() { return data; }
    CK_HOST_DEVICE const auto& get() const { return data; }
    CK_HOST_DEVICE auto& get(index_t i) { return data[i]; }
    CK_HOST_DEVICE const auto& get(index_t i) const { return data[i]; }

    template <index_t I>
    CK_HOST_DEVICE auto& operator[](number<I>)
    {
        return data[I];
    }
    template <index_t I>
    CK_HOST_DEVICE const auto& operator[](number<I>) const
    {
        return data[I];
    }
    template <index_t I>
    CK_HOST_DEVICE auto& operator()(number<I>)
    {
        return data[I];
    }

    CK_HOST_DEVICE auto& at(index_t i) { return data[i]; }
    CK_HOST_DEVICE const auto& at(index_t i) const { return data[i]; }
    template <index_t I>
    CK_HOST_DEVICE auto& at()
    {
        return data[I];
    }
    template <index_t I>
    CK_HOST_DEVICE const auto& at() const
    {
        return data[I];
    }
    template <index_t I>
    CK_HOST_DEVICE auto& at(number<I>)
    {
        return data[I];
    }
    template <index_t I>
    CK_HOST_DEVICE const auto& at(number<I>) const
    {
        return data[I];
    }

#define _VT_COMMON_AS()                                      \
    static_assert(sizeof(value_type) * N % sizeof(Tx) == 0); \
    constexpr int vx = sizeof(value_type) * N / sizeof(Tx)

    template <typename Tx>
    CK_HOST_DEVICE auto& get_as()
    {
        _VT_COMMON_AS();
        return reinterpret_cast<array<Tx, vx>&>(data);
    }
    template <typename Tx>
    CK_HOST_DEVICE const auto& get_as() const
    {
        _VT_COMMON_AS();
        return reinterpret_cast<const array<Tx, vx>&>(data);
    }
    template <typename Tx>
    CK_HOST_DEVICE auto& get_as(index_t i)
    {
        _VT_COMMON_AS();
        return reinterpret_cast<array<Tx, vx>&>(data).get(i);
    }
    template <typename Tx>
    CK_HOST_DEVICE const auto& get_as(index_t i) const
    {
        _VT_COMMON_AS();
        return reinterpret_cast<const array<Tx, vx>&>(data).get(i);
    }
#undef _VT_COMMON_AS
};

template <typename T, index_t N>
struct vector_type_maker
{
    using type = vector_type<T, N>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<T __attribute__((ext_vector_type(N1))), N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<vector_type<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N>
using vector_type_maker_t = typename vector_type_maker<T, N>::type;

template <typename T, index_t N>
CK_HOST_DEVICE constexpr auto make_vector_type(number<N>)
{
    return typename vector_type_maker<T, N>::type{};
}

// scalar_type
template <typename TV>
struct scalar_type;

// is_scalar_type
template <typename TV>
struct is_scalar_type
{
    static constexpr bool value = (scalar_type<remove_cvref_t<TV>>::vector_size == 1);
};

// has_same_scalar_type
template <typename X, typename Y>
using has_same_scalar_type = is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                     typename scalar_type<remove_cvref_t<Y>>::type>;

template <typename T, index_t N>
struct scalar_type<T __attribute__((ext_vector_type(N)))>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct scalar_type<vector_type<T, N>>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

//
template <>
struct scalar_type<double>
{
    using type                           = double;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<float>
{
    using type                           = float;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<half_t>
{
    using type                           = half_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bhalf_t>
{
    using type                           = bhalf_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int64_t>
{
    using type                           = int64_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int32_t>
{
    using type                           = int32_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int8_t>
{
    using type                           = int8_t;
    static constexpr index_t vector_size = 1;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct scalar_type<int4_t>
{
    using type                           = int4_t;
    static constexpr index_t vector_size = 1;
};
#endif

template <>
struct scalar_type<fp8_t>
{
    using type                           = fp8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_t>
{
    using type                           = bf8_t;
    static constexpr index_t vector_size = 1;
};

// below are some pre-defines of ext_vector_type
// fp64
using double2_t = typename vector_type<double, 2>::type;
using double4_t = typename vector_type<double, 4>::type;

// fp32
using float2_t  = typename vector_type<float, 2>::type;
using float4_t  = typename vector_type<float, 4>::type;
using float8_t  = typename vector_type<float, 8>::type;
using float16_t = typename vector_type<float, 16>::type;
using float32_t = typename vector_type<float, 32>::type;
using float64_t = typename vector_type<float, 64>::type;

// fp16
using half2_t  = typename vector_type<half_t, 2>::type;
using half4_t  = typename vector_type<half_t, 4>::type;
using half8_t  = typename vector_type<half_t, 8>::type;
using half16_t = typename vector_type<half_t, 16>::type;
using half32_t = typename vector_type<half_t, 32>::type;
using half64_t = typename vector_type<half_t, 64>::type;

// bfp16
using bhalf2_t  = typename vector_type<bhalf_t, 2>::type;
using bhalf4_t  = typename vector_type<bhalf_t, 4>::type;
using bhalf8_t  = typename vector_type<bhalf_t, 8>::type;
using bhalf16_t = typename vector_type<bhalf_t, 16>::type;
using bhalf32_t = typename vector_type<bhalf_t, 32>::type;
using bhalf64_t = typename vector_type<bhalf_t, 64>::type;

// i32
using int32x2_t  = typename vector_type<int32_t, 2>::type;
using int32x4_t  = typename vector_type<int32_t, 4>::type;
using int32x8_t  = typename vector_type<int32_t, 8>::type;
using int32x16_t = typename vector_type<int32_t, 16>::type;
using int32x32_t = typename vector_type<int32_t, 32>::type;
using int32x64_t = typename vector_type<int32_t, 64>::type;

// i8
using int8x2_t  = typename vector_type<int8_t, 2>::type;
using int8x4_t  = typename vector_type<int8_t, 4>::type;
using int8x8_t  = typename vector_type<int8_t, 8>::type;
using int8x16_t = typename vector_type<int8_t, 16>::type;
using int8x32_t = typename vector_type<int8_t, 32>::type;
using int8x64_t = typename vector_type<int8_t, 64>::type;

// f8
using fp8x2_t  = typename vector_type<fp8_t, 2>::type;
using fp8x4_t  = typename vector_type<fp8_t, 4>::type;
using fp8x8_t  = typename vector_type<fp8_t, 8>::type;
using fp8x16_t = typename vector_type<fp8_t, 16>::type;
using fp8x32_t = typename vector_type<fp8_t, 32>::type;
using fp8x64_t = typename vector_type<fp8_t, 64>::type;

// bf8
using bf8x2_t  = typename vector_type<bf8_t, 2>::type;
using bf8x4_t  = typename vector_type<bf8_t, 4>::type;
using bf8x8_t  = typename vector_type<bf8_t, 8>::type;
using bf8x16_t = typename vector_type<bf8_t, 16>::type;
using bf8x32_t = typename vector_type<bf8_t, 32>::type;
using bf8x64_t = typename vector_type<bf8_t, 64>::type;

} // namespace ck_tile
