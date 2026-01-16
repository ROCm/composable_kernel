// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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

// Functor for merge_sequences to avoid lambda instantiation overhead
struct merge_sequences_functor
{
    template <typename... Seqs>
    __host__ __device__ constexpr auto operator()(Seqs... seqs) const
    {
        return merge_sequences(seqs...);
    }
};

// Helper to unpack a tuple of sequences and merge them
// Replaces: unpack([](auto... xs) { return merge_sequences(xs...); }, tuple_of_sequences)
template <typename TupleOfSequences>
__host__ __device__ constexpr auto unpack_and_merge_sequences(TupleOfSequences)
{
    return unpack(merge_sequences_functor{}, TupleOfSequences{});
}

} // namespace ck
