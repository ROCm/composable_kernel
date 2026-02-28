// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/functional4.hpp"
#include "ck/utility/tuple.hpp"

namespace ck {

template <index_t... Is>
__host__ __device__ constexpr auto make_sequence(Number<Is>...)
{
    return Sequence<Is...>{};
}

// F returns index_t
template <typename F, index_t N>
__host__ __device__ constexpr auto generate_sequence(F, Number<N>)
{
    return typename sequence_gen<N, F>::type{};
}

// F returns Number<>
template <typename F, index_t N>
__host__ __device__ constexpr auto generate_sequence_v2(F&& f, Number<N>)
{
    return unpack([&f](auto&&... xs) { return make_sequence(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <index_t... Is>
__host__ __device__ constexpr auto to_sequence(Tuple<Number<Is>...>)
{
    return Sequence<Is...>{};
}

// Functor wrapper for merge_sequences to enable reuse across call sites
struct merge_sequences_functor
{
    template <typename... Seqs>
    __host__ __device__ constexpr auto operator()(Seqs... seqs) const
    {
        return merge_sequences(seqs...);
    }
};

// Unpacks tuple of sequences and merges them into a single sequence
template <typename TupleOfSequences>
__host__ __device__ constexpr auto unpack_and_merge_sequences(TupleOfSequences tuple_of_sequences)
{
    return unpack(merge_sequences_functor{}, tuple_of_sequences);
}

} // namespace ck
