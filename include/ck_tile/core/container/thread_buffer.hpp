// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/tuple.hpp"

namespace ck_tile {

#if CK_TILE_THREAD_BUFFER_DEFAULT == CK_TILE_THREAD_BUFFER_USE_TUPLE
template <typename T, index_t N>
using thread_buffer = tuple_array<T, N>;

template <typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_thread_buffer(Ts&&... ts)
{
    return make_tuple(ts...);
}
#else

#if 0
template <typename T, index_t N>
using thread_buffer = array<T, N>;

template <typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_thread_buffer(Ts&&... ts)
{
    return make_array(ts...);
}

#endif

// clang-format off
template<typename T_, index_t N_>
struct thread_buffer {
    using value_type = remove_cvref_t<T_>;
    static constexpr index_t N = N_;

    value_type data[N];

    // TODO: this ctor can't ignore
    CK_TILE_HOST_DEVICE constexpr thread_buffer() : data{} {}
    CK_TILE_HOST_DEVICE constexpr thread_buffer(const value_type & o) : data{o} {}

    CK_TILE_HOST_DEVICE static constexpr auto size() { return N; }
    CK_TILE_HOST_DEVICE auto & get() {return data; }
    CK_TILE_HOST_DEVICE const auto & get() const {return data; }
    CK_TILE_HOST_DEVICE auto & get(index_t i) {return data[i]; }
    CK_TILE_HOST_DEVICE const auto & get(index_t i) const {return data[i]; }
    CK_TILE_HOST_DEVICE constexpr const auto& operator[](index_t i) const { return get(i); }
    CK_TILE_HOST_DEVICE constexpr auto& operator[](index_t i)             { return get(i); }
    CK_TILE_HOST_DEVICE constexpr auto& operator()(index_t i)             { return get(i); }     // TODO: compatible
    CK_TILE_HOST_DEVICE constexpr auto& at(index_t i)                                   { return get(i); }
    CK_TILE_HOST_DEVICE constexpr const auto& at(index_t i) const                       { return get(i); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr auto& at()                       { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr const auto& at() const           { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr auto& at(number<I>)              { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr const auto& at(number<I>) const  { return get(I); }

#define TB_COMMON_AS() \
            static_assert(sizeof(value_type) * N % sizeof(Tx) == 0); \
            constexpr int vx = sizeof(value_type) * N / sizeof(Tx)

    template<typename Tx>
    CK_TILE_HOST_DEVICE auto & get_as() {TB_COMMON_AS();
            return reinterpret_cast<thread_buffer<Tx, vx>&>(data);}
    template<typename Tx>
    CK_TILE_HOST_DEVICE const auto & get_as() const {TB_COMMON_AS();
            return reinterpret_cast<const thread_buffer<Tx, vx>&>(data);}
    template<typename Tx>
    CK_TILE_HOST_DEVICE auto & get_as(index_t i) {TB_COMMON_AS();
            return reinterpret_cast<thread_buffer<Tx, vx>&>(data).get(i);}
    template<typename Tx>
    CK_TILE_HOST_DEVICE const auto & get_as(index_t i) const {TB_COMMON_AS();
            return reinterpret_cast<const thread_buffer<Tx, vx>&>(data).get(i);}

    template <typename Tx> CK_TILE_HOST_DEVICE constexpr void set_as(index_t i, const Tx & x)
            { TB_COMMON_AS();    reinterpret_cast<array<Tx, vx>&>(data).at(i) = x; }
    template <typename Tx, index_t I> CK_TILE_HOST_DEVICE constexpr void set_as(number<I>, const Tx & x)
            { TB_COMMON_AS();    reinterpret_cast<array<Tx, vx>&>(data).at(number<I>{}) = x; }
#undef TB_COMMON_AS
};
// clang-format on

template <typename>
struct vector_traits;

// specialization for array
template <typename T, index_t N>
struct vector_traits<thread_buffer<T, N>>
{
    using scalar_type                    = T;
    static constexpr index_t vector_size = N;
};

#endif

} // namespace ck_tile
