// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "sequence.hpp"
#include "array.hpp"
#include "tuple.hpp"

namespace ck {

template <typename... Seqs>
__host__ __device__ constexpr auto to_array_of_array(Tuple<Seqs...> t_of_s)
{
    constexpr index_t n0 = sizeof...(Seqs);

    constexpr index_t max_n1 = [&] {
        index_t max_n1_ = 0;

        static_for<0, n0, 1>{}([&](auto i0) {
            constexpr index_t n1 = t_of_s[i0].Size();

            max_n1_ = max_n1_ < n1 ? n1 : max_n1_;
        });

        return max_n1_;
    }();

    Array<Array<index_t, max_n1>, n0> a_of_a{{-1}};

    static_for<0, n0, 1>{}([&](auto i0) {
        constexpr index_t n1 = t_of_s[i0].Size();

        static_for<0, n1, 1>{}([&](auto i1) { a_of_a(i0)(i1) = t_of_s[i0][i1]; });
    });

    return a_of_a;
}

} // namespace ck
