// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iterator>
#include <numeric>

namespace ck {
template <typename T, typename ForwardIterator, typename Size, typename BinaryOperation>
__host__ __device__ auto accumulate_n(ForwardIterator first, Size count, T init, BinaryOperation op)
    -> decltype(std::accumulate(first, std::next(first, count), init, op))
{
    return std::accumulate(first, std::next(first, count), init, op);
}

template <typename T, typename ForwardIterator, typename Size>
__host__ __device__ auto mult_accumulate_n(ForwardIterator first, Size count, T init)
{
    for(ForwardIterator x = first; x != first + count; x++)
    {
        init *= *x;
    }
    return init;
}
} // namespace ck
