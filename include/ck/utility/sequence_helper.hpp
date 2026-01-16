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

// O(1) template depth implementation using pack expansion
// Avoids O(N) recursive template instantiations
template <index_t Target, typename... Seqs>
struct FindInTupleOfSequencesCompute
{
    private:
    // Result struct for constexpr computation
    struct ResultData
    {
        index_t itran;
        index_t idim_up;
        bool found;
    };

    // Compute result using constexpr function with array lookup
    static constexpr ResultData compute()
    {
        if constexpr(sizeof...(Seqs) == 0)
        {
            return {0, 0, false};
        }
        else
        {
            // Pack expansion creates array - O(1) template depth
            constexpr index_t indices[] = {sequence_find_value<Target>(Seqs{})...};

            // Find first matching sequence
            for(index_t i = 0; i < static_cast<index_t>(sizeof...(Seqs)); ++i)
            {
                if(indices[i] >= 0)
                {
                    return {i, indices[i], true};
                }
            }
            return {0, 0, false};
        }
    }

    static constexpr ResultData result_ = compute();

    public:
    static constexpr index_t itran   = result_.itran;
    static constexpr index_t idim_up = result_.idim_up;
    static constexpr bool found      = result_.found;

    using type = FindTransformResult<itran, idim_up, found>;
};

// Find target value in a tuple of sequences
// Returns FindTransformResult<itran, idim_up, found>
// Uses O(1) template depth via pack expansion (no recursion)
template <index_t Target, typename... Seqs>
__host__ __device__ constexpr auto find_in_tuple_of_sequences(Tuple<Seqs...>)
{
    return typename FindInTupleOfSequencesCompute<Target, Seqs...>::type{};
}

} // namespace ck
