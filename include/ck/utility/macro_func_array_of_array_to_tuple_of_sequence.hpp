// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/macro_func_array_to_sequence.hpp"

// Macro function
// convert constexpr Array<Array<index_t, xxx>, xxx> to Tuple<Sequence<...>, ...>
//   Input:
//     1. a_of_b_impl: constexpr Array<Array<index_t, xxx>, xxx>
//     2. a_size: constexper index_t
//     3. bs_sizes: constexpr Array<index_t, xxx>
//   Output:
//     Tuple<Sequence<...>, ...>
#define TO_TUPLE_OF_SEQUENCE(a_of_b_impl, a_size, bs_sizes)             \
    [a_of_b_impl, a_size, bs_sizes] {                                   \
        return ck::generate_tuple(                                      \
            [=](auto i) {                                               \
                constexpr auto b_impl    = a_of_b_impl[i];              \
                constexpr index_t b_size = bs_sizes[i];                 \
                constexpr auto b         = TO_SEQUENCE(b_impl, b_size); \
                return b;                                               \
            },                                                          \
            ck::Number<a_size>{});                                      \
    }()
