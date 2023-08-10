// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/inner_product_dpp8.hpp"

namespace ck {

namespace dpp8 {

template <index_t MPerLanegroup,
          index_t NPerLanegroup,
          index_t KPerLanegroup,
          class FloatA,
          class FloatB,
          class FloatVecC,
          bool ShareA>
__device__ void RunGemm(const FloatA& a, const FloatB& b, FloatVecC& c_vec)
{
    constexpr index_t c_dim = ShareA ? MPerLanegroup : NPerLanegroup;

    const vector_type<half_t, KPerLanegroup> a_vector{a};
    const vector_type<half_t, KPerLanegroup> b_vector{b};

    static_for<0, c_dim, 1>{}([&](auto c_idx) {
        float c = c_vec.template AsType<float>()(c_idx);
        // Next `c_idx` implies that we need to pull data from the next lane.
        constexpr index_t source_lane = c_idx;
        static_for<0, KPerLanegroup / 2, 1>{}([&](auto k_chunk) {
            const auto a_half2 = a_vector.template AsType<half2_t>()[k_chunk];
            const auto b_half2 = b_vector.template AsType<half2_t>()[k_chunk];
            ck::dpp8::inner_product_dpp<half2_t, half2_t, float, source_lane, ShareA>(
                a_half2, b_half2, c);
        });
        c_vec.template AsType<float>()(c_idx) = c;
    });
}

} // namespace dpp8

} // namespace ck
