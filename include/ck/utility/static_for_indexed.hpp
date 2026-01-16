// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/sequence.hpp"
#include "ck/utility/number.hpp"

namespace ck {

namespace detail {

// Helper to call functor with compile-time index as template parameter
// This avoids per-lambda instantiation by using a struct with templated operator()
template <typename F, index_t I>
__host__ __device__ __forceinline__ constexpr void call_at_index(F& f)
{
    f.template operator()<I>();
}

template <typename Seq>
struct static_for_indexed_impl;

template <index_t... Is>
struct static_for_indexed_impl<Sequence<Is...>>
{
    template <typename F>
    __host__ __device__ constexpr void operator()(F& f) const
    {
        (call_at_index<F, Is>(f), ...);
    }

    template <typename F>
    __host__ __device__ constexpr void operator()(const F& f) const
    {
        (call_at_index<const F, Is>(const_cast<const F&>(f)), ...);
    }
};

} // namespace detail

// static_for_indexed: Optimized compile-time loop that reduces template instantiation bloat
//
// Unlike static_for which takes a lambda (creating unique instantiation per lambda type),
// static_for_indexed expects a functor with a templated operator():
//
//   struct MyLoop {
//       SomeType& data;
//       template <index_t I>
//       __host__ __device__ void operator()() const { data[Number<I>{}] = I * 2; }
//   };
//   static_for_indexed<0, N, 1>{}(MyLoop{data});
//
// This reduces instantiations because MyLoop is a named type reused across call sites,
// rather than unique lambda types at each usage.
//
template <index_t NBegin, index_t NEnd, index_t Increment = 1>
struct static_for_indexed
{
    template <class F>
    __host__ __device__ constexpr void operator()(F& f) const
    {
        detail::static_for_indexed_impl<
            typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(f);
    }

    template <class F>
    __host__ __device__ constexpr void operator()(const F& f) const
    {
        detail::static_for_indexed_impl<
            typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(f);
    }
};

// Convenience macro for inline loop body definition
// Usage:
//   CK_STATIC_FOR_INDEXED(0, 8, 1, I, {
//       arr[Number<I>{}] = I * 2;
//   });
//
// Note: Captures must be handled via the struct members, not lambda-style captures
#define CK_STATIC_FOR_INDEXED_IMPL2(begin, end, incr, idx, captures, body) \
    do                                                                     \
    {                                                                      \
        struct CK_LocalLoop                                                \
        {                                                                  \
            captures template <ck::index_t idx>                            \
            __host__ __device__ void operator()() const body               \
        } ck_loop_body;                                                    \
        ck::static_for_indexed<begin, end, incr>{}(ck_loop_body);          \
    } while(0)

// Simpler macro without explicit captures (for self-contained loop bodies)
#define CK_STATIC_FOR_INDEXED(begin, end, incr, idx, body) \
    CK_STATIC_FOR_INDEXED_IMPL2(begin, end, incr, idx, , body)

} // namespace ck
