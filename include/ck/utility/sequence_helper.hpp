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

// sequence_find_value - O(1) template depth constexpr search
//
// Optimization: Constexpr loop with array lookup instead of recursive template pattern
//
// Why this approach:
// - Recursive template (OLD): template instantiation for each recursion level -> O(N)
// instantiations
//   Example: Finding value in Sequence<1,2,3,4,5> requires 5 recursive instantiations
//
// - Constexpr loop (NEW): Single function instantiation with runtime loop -> O(1) instantiation
//   Same search requires only 1 function instantiation, loop executes at compile-time
//
// Implementation details:
//   1. Pack expansion creates constexpr array: {(Is == Target)...}
//   2. Constexpr for loop searches the array
//   3. Entire function evaluates at compile-time (no runtime cost)
//
// Impact:
// - Significantly reduces template instantiation depth for sequence search operations
// - Dramatically improves compilation time vs recursive template approach
// - Pattern applies to any compile-time search/lookup operation
//
// Trade-off: Uses constexpr evaluation instead of pure template metaprogramming.
// Requires C++14 constexpr but results in dramatically better compile times.
//
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

// find_in_tuple_of_sequences - finds which sequence contains a target value
//
// Optimization: Pack expansion with constexpr search instead of nested static_for loops
//
// Why this approach:
// - Nested static_for (OLD): Creates lambda closure for each iteration level
//   Example: Searching Tuple<Seq<0,1>, Seq<2,3>, Seq<4,5>> creates multiple applier::operator()
//   instantiations. Result: Many applier instantiations for typical tensor descriptor operations.
//
// - Pack expansion + constexpr (NEW): Single function with compile-time array search
//   Example: Same search creates constexpr array, single search function.
//   Result: 1 function instantiation regardless of tuple size.
//
// Implementation:
//   1. Pack expansion: sequence_find_value<Target>(Seqs{})... applies search to each sequence
//   2. Results collected in constexpr array
//   3. Linear search finds first non-negative result (sequence containing target)
//
// Impact:
// - Significantly reduces applier::operator() instantiations in tensor descriptor transforms
// - O(1) template depth instead of O(N*M) for N sequences of length M
//
// Use case: Finding which dimension index contains a specific value (common in tensor reordering)
//
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
