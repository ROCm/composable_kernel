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

// Find index of Target in Sequence, returns -1 if not found
// Uses constexpr array lookup for O(1) template depth
template <index_t Target, index_t... Is>
__host__ __device__ constexpr index_t sequence_find_value(Sequence<Is...>)
{
    if constexpr(sizeof...(Is) == 0)
    {
        return -1;
    }
    else
    {
        constexpr bool matches[] = {(Is == Target)...};
        for(index_t i = 0; i < static_cast<index_t>(sizeof...(Is)); ++i)
        {
            if(matches[i])
                return i;
        }
        return -1;
    }
}

// Result type for find_in_tuple_of_sequences
template <index_t ITran, index_t IDimUp, bool Found>
struct FindTransformResult
{
    static constexpr index_t itran   = ITran;
    static constexpr index_t idim_up = IDimUp;
    static constexpr bool found      = Found;
};

namespace detail {

// Helper to search through a tuple of sequences for a target value
// Returns FindTransformResult with (transform_index, index_within_sequence, found)
template <index_t Target, index_t TranIdx, typename FirstSeq, typename... RestSeqs>
__host__ __device__ constexpr auto find_in_tuple_of_sequences_impl()
{
    constexpr index_t idx = sequence_find_value<Target>(FirstSeq{});
    if constexpr(idx >= 0)
    {
        return FindTransformResult<TranIdx, idx, true>{};
    }
    else if constexpr(sizeof...(RestSeqs) > 0)
    {
        return find_in_tuple_of_sequences_impl<Target, TranIdx + 1, RestSeqs...>();
    }
    else
    {
        return FindTransformResult<0, 0, false>{};
    }
}

} // namespace detail

// Find target value in a tuple of sequences
// Returns FindTransformResult<itran, idim_up, found>
// This replaces nested static_for loops with O(1) template depth
template <index_t Target, typename... Seqs>
__host__ __device__ constexpr auto find_in_tuple_of_sequences(Tuple<Seqs...>)
{
    if constexpr(sizeof...(Seqs) == 0)
    {
        return FindTransformResult<0, 0, false>{};
    }
    else
    {
        return detail::find_in_tuple_of_sequences_impl<Target, 0, Seqs...>();
    }
}

} // namespace ck
