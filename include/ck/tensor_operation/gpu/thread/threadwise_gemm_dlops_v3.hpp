// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_THREADWISE_GEMM_DLOPS_V3_HPP
#define CK_THREADWISE_GEMM_DLOPS_V3_HPP

#include "ck/utility/common_header.hpp"

namespace ck {

// C[M, N] += transpose(A[M, M]) * B[M, N]
//   Element of matrix can be vectorized data
template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AThreadDesc_K0_M_K1,
          typename BThreadDesc_K0_N_K1,
          typename CThreadDesc_M_N,
          typename enable_if<AThreadDesc_K0_M_K1::IsKnownAtCompileTime() &&
                                 BThreadDesc_K0_N_K1::IsKnownAtCompileTime() &&
                                 CThreadDesc_M_N::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseGemmDlops_km_kn_mn_v3
{

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

        static_assert(AThreadDesc_K0_M_K1::IsKnownAtCompileTime() &&
                          BThreadDesc_K0_N_K1::IsKnownAtCompileTime() &&
                          CThreadDesc_M_N::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<AOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<BOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<COriginIdx>>::value,
                      "wrong! AOriginIdx, BOriginIdx, COringinIdx should be known at compile-time");

        static_assert(
            is_same<remove_cvref_t<typename ABuffer::type>, remove_cvref_t<FloatA>>::value &&
            is_same<remove_cvref_t<typename BBuffer::type>, remove_cvref_t<FloatB>>::value &&
            is_same<remove_cvref_t<typename CBuffer::type>, remove_cvref_t<FloatC>>::value &&
            "wrong! inconsistent type");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        constexpr auto K0 = AThreadDesc_K0_M_K1{}.GetLength(I0);
        constexpr auto M  = AThreadDesc_K0_M_K1{}.GetLength(I1);
        constexpr auto K1 = AThreadDesc_K0_M_K1{}.GetLength(I2);

        constexpr auto N = BThreadDesc_K0_N_K1{}.GetLength(I1);

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        static_for<0, M, 1>{}([&](auto m) {
            static_for<0, N, 1>{}([&](auto n) {
                static_for<0, K0, 1>{}([&](auto k0) {
                    static_for<0, K1, 1>{}([&](auto k1) {
                        constexpr index_t a_offset = AThreadDesc_K0_M_K1{}.CalculateOffset(
                            a_origin_idx + make_tuple(k0, m, k1));

                        constexpr index_t b_offset = BThreadDesc_K0_N_K1{}.CalculateOffset(
                            b_origin_idx + make_tuple(k0, n, k1));

                        constexpr index_t c_offset = CThreadDesc_M_N{}.CalculateOffset(
                            c_origin_idx + make_tuple(0, m, 0, n));

                        inner_product<FloatA, FloatB, FloatC>(a_buf[Number<a_offset>{}],
                                                              b_buf[Number<b_offset>{}],
                                                              c_buf(Number<c_offset>{}));
                    });
                });
            });
        });
    } // namespace ck
};

} // namespace ck
#endif
