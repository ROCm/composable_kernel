// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/amd_gemm_dpp.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/inner_product_dpp8.hpp"
#include "ck/utility/math.hpp"

namespace ck {

/**
 * Threadwise contraction using dot instructions with DPP8 modifier.
 *
 * Assumptions:
 *   1. `AThreadDesc_TK0_TM0_TM1_TK1`, `BThreadDesc_TK0_TN0_TN1_TK1`, `CThreadDesc_TM0_TM1_TN0_TN1`
 *      are known at compile-time;
 *   2. `AOriginIdx`, `BOriginIdx`, `COriginIdx` are known at compile-time;
 *   3. `TM0` is equal to 1 and `TN0` is equal to 1;
 *   4. When `ShareA` is set (unset, respectively), `TM1` (`TN1`, respectively) is divisible by
 *      the size of the lane group (`dpp8::lane_group_size`).
 */
template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AThreadDesc_TK0_TM0_TM1_TK1,
          typename BThreadDesc_TK0_TN0_TN1_TK1,
          typename CThreadDesc_TM0_TM1_TN0_TN1,
          typename TKLengths,
          typename TMLengths,
          typename TNLengths,
          bool ShareA,
          typename enable_if<AThreadDesc_TK0_TM0_TM1_TK1::IsKnownAtCompileTime() &&
                                 BThreadDesc_TK0_TN0_TN1_TK1::IsKnownAtCompileTime() &&
                                 CThreadDesc_TM0_TM1_TN0_TN1::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseContractionDlDpp8_A_TK0_TM0_TM1_TK1_B_TK0_TN0_TN1_TK1_C_TM0_TM1_TN0_TN1
{

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t TK0 = TKLengths{}[I0];
    static constexpr index_t TK1 = TKLengths{}[I1];
    static constexpr index_t TM0 = TMLengths{}[I0];
    static constexpr index_t TM1 = TMLengths{}[I1];
    static constexpr index_t TN0 = TNLengths{}[I0];
    static constexpr index_t TN1 = TNLengths{}[I1];

    static_assert(TM0 == 1 && TN0 == 1);

    static_assert((ShareA && TM1 % dpp8::lane_group_size == 0) ||
                  (!ShareA && TN1 % dpp8::lane_group_size == 0));
    static constexpr index_t shared_elems_per_lane =
        ShareA ? TM1 / dpp8::lane_group_size : TN1 / dpp8::lane_group_size;

    __device__ constexpr ThreadwiseContractionDlDpp8_A_TK0_TM0_TM1_TK1_B_TK0_TN0_TN1_TK1_C_TM0_TM1_TN0_TN1()
    {
        static_assert(AThreadDesc_TK0_TM0_TM1_TK1::IsKnownAtCompileTime() &&
                          BThreadDesc_TK0_TN0_TN1_TK1::IsKnownAtCompileTime() &&
                          CThreadDesc_TM0_TM1_TN0_TN1::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(TKLengths::Size() == 2 && TMLengths::Size() == 2 && TNLengths::Size() == 2,
                      "wrong!");
    }

    template <typename ABuffer,
              typename AOriginIdx,
              typename BBuffer,
              typename BOriginIdx,
              typename CBuffer,
              typename COriginIdx>
    __device__ static void Run(const ABuffer& a_buf,
                               AOriginIdx,
                               const BBuffer& b_buf,
                               BOriginIdx,
                               CBuffer& c_buf,
                               COriginIdx)
    {
        static_assert(is_known_at_compile_time<remove_cvref_t<AOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<BOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<COriginIdx>>::value,
                      "wrong! AOriginIdx, BOriginIdx, COringinIdx should be known at compile-time");

        static_assert(
            is_same<remove_cvref_t<typename ABuffer::type>, remove_cvref_t<FloatA>>::value &&
            is_same<remove_cvref_t<typename BBuffer::type>, remove_cvref_t<FloatB>>::value &&
            is_same<remove_cvref_t<typename CBuffer::type>, remove_cvref_t<FloatC>>::value &&
            "wrong! inconsistent type");

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        static_for<0, TK0, 1>{}([&](auto tk0) {
            static_for<0, TM1, 1>{}([&](auto tm1) {
                static_for<0, TN1, 1>{}([&](auto tn1) {
                    vector_type<FloatA, TK1> a_vec;
                    vector_type<FloatB, TK1> b_vec;

                    static_for<0, TK1, 1>{}([&](auto tk1) {
                        constexpr index_t local_tm1 = ShareA ? tm1 % shared_elems_per_lane : tm1;
                        constexpr index_t a_offset  = AThreadDesc_TK0_TM0_TM1_TK1{}.CalculateOffset(
                            a_origin_idx + make_multi_index(tk0, 0, local_tm1, tk1));

                        constexpr index_t local_tn1 = ShareA ? tn1 : tn1 % shared_elems_per_lane;
                        constexpr index_t b_offset  = BThreadDesc_TK0_TN0_TN1_TK1{}.CalculateOffset(
                            b_origin_idx + make_multi_index(tk0, 0, local_tn1, tk1));

                        a_vec.template AsType<FloatA>()(tk1) = a_buf[Number<a_offset>{}];
                        b_vec.template AsType<FloatB>()(tk1) = b_buf[Number<b_offset>{}];
                    });

                    using a_vector_t = typename vector_type<FloatA, TK1>::type;
                    using b_vector_t = typename vector_type<FloatB, TK1>::type;

                    constexpr index_t c_offset = CThreadDesc_TM0_TM1_TN0_TN1{}.CalculateOffset(
                        c_origin_idx + make_multi_index(0, tm1, 0, tn1));

                    constexpr int src_lane =
                        ShareA ? (tm1 / shared_elems_per_lane) % dpp8::lane_group_size
                               : (tn1 / shared_elems_per_lane) % dpp8::lane_group_size;

                    dpp8::inner_product_dpp<a_vector_t, b_vector_t, FloatC, src_lane, ShareA>(
                        a_vec.template AsType<a_vector_t>()[I0],
                        b_vec.template AsType<b_vector_t>()[I0],
                        c_buf(Number<c_offset>{}));
                });
            });
        });
    }
};

} // namespace ck
