// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/sequence.hpp"
#include "ck/utility/array.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/macro_func_array_to_sequence.hpp"

namespace ck {

template <index_t... Is>
__host__ __device__ constexpr auto make_sequence(Number<Is>...)
{
    return Sequence<Is...>{};
}

// F() returns index_t
// F use default constructor, so F cannot be lambda function
template <typename F, index_t N>
__host__ __device__ constexpr auto generate_sequence(F, Number<N>)
{
    return typename sequence_gen<N, F>::type{};
}

// F() returns Number<>
// F could be lambda function
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

namespace detail {
template <index_t h_idx, typename SeqSortedSamples, typename SeqRange>
struct sorted_sequence_histogram;

template <index_t h_idx, index_t x, index_t... xs, index_t r, index_t... rs>
struct sorted_sequence_histogram<h_idx, Sequence<x, xs...>, Sequence<r, rs...>>
{
    template <typename Histogram>
    constexpr auto operator()(Histogram& h)
    {
        if constexpr(x < r)
        {
            h.template At<h_idx>() += 1;
            sorted_sequence_histogram<h_idx, Sequence<xs...>, Sequence<r, rs...>>{}(h);
        }
        else
        {
            h.template At<h_idx + 1>() = 1;
            sorted_sequence_histogram<h_idx + 1, Sequence<xs...>, Sequence<rs...>>{}(h);
        }
    }
};

template <index_t h_idx, index_t x, index_t r, index_t... rs>
struct sorted_sequence_histogram<h_idx, Sequence<x>, Sequence<r, rs...>>
{
    template <typename Histogram>
    constexpr auto operator()(Histogram& h)
    {
        if constexpr(x < r)
        {
            h.template At<h_idx>() += 1;
        }
    }
};
} // namespace detail

// SeqSortedSamples: <0, 2, 3, 5, 7>, SeqRange: <0, 3, 6, 9> -> SeqHistogram : <2, 2, 1>
template <typename SeqSortedSamples, index_t r, index_t... rs>
constexpr auto histogram_sorted_sequence(SeqSortedSamples, Sequence<r, rs...>)
{
    constexpr auto bins      = sizeof...(rs); // or categories
    constexpr auto histogram = [&]() {
        Array<index_t, bins> h{0}; // make sure this can clear all element to zero
        detail::sorted_sequence_histogram<0, SeqSortedSamples, Sequence<rs...>>{}(h);
        return h;
    }();

    return TO_SEQUENCE(histogram, bins);
}

} // namespace ck
