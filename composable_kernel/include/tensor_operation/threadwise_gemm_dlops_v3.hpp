#ifndef CK_THREADWISE_GEMM_DLOPS_V3_HPP
#define CK_THREADWISE_GEMM_DLOPS_V3_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {

// C[M, N] += transpose(A[K, M]) * B[K, N]
//   Element of matrix can be vectorized data
// Assume:
//   1. AThreadDesc_E1_K_E2, BThreadDesc_E1_N_Ho_Wo_E2, CThreadDesc_K_N_Ho_Wo are known at
//   compile-time
//   2. AOriginIdx, BOriginIdx, COriginIdx are known at compile-time
template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AThreadDesc_E1_K_E2,
          typename BThreadDesc_E1_N_Ho_Wo_E2,
          typename CThreadDesc_K_N_Ho_Wo,
          typename enable_if<AThreadDesc_E1_K_E2::IsKnownAtCompileTime() &&
                                 BThreadDesc_E1_N_Ho_Wo_E2::IsKnownAtCompileTime() &&
                                 CThreadDesc_K_N_Ho_Wo::IsKnownAtCompileTime(),
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

        static_assert(AThreadDesc_E1_K_E2::IsKnownAtCompileTime() &&
                          BThreadDesc_E1_N_Ho_Wo_E2::IsKnownAtCompileTime() &&
                          CThreadDesc_K_N_Ho_Wo::IsKnownAtCompileTime(),
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
        constexpr auto I3 = Number<3>{};

        constexpr auto E1 = AThreadDesc_E1_K_E2{}.GetLength(I0);
        constexpr auto K  = AThreadDesc_E1_K_E2{}.GetLength(I1);
        constexpr auto E2 = AThreadDesc_E1_K_E2{}.GetLength(I2);

        static_assert(E1 == 4 && E2 == 4, "");

        constexpr auto H = BThreadDesc_E1_N_Ho_Wo_E2{}.GetLength(I2);
        constexpr auto W = BThreadDesc_E1_N_Ho_Wo_E2{}.GetLength(I3);

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        static_for<0, K, 1>{}([&](auto k) {
            static_for<0, H, 1>{}([&](auto h) {
                static_for<0, W, 1>{}([&](auto w) {
                    static_for<0, E1, 1>{}([&](auto e) {
                        vector_type<FloatA, E2> a_vec;
                        vector_type<FloatB, E2> b_vec;

                        static_for<0, E2, 1>{}([&](auto v) {
                            constexpr index_t a_offset = AThreadDesc_E1_K_E2{}.CalculateOffset(
                                a_origin_idx + make_tuple(e, k, v));
                            constexpr index_t b_offset =
                                BThreadDesc_E1_N_Ho_Wo_E2{}.CalculateOffset(
                                    b_origin_idx + make_tuple(e, 0, h, w, v));

                            a_vec.template AsType<FloatA>()(v) = a_buf[Number<a_offset>{}];
                            b_vec.template AsType<FloatB>()(v) = b_buf[Number<b_offset>{}];
                        });

                        using a_vector_t = typename vector_type<FloatA, E2>::type;
                        using b_vector_t = typename vector_type<FloatB, E2>::type;

                        constexpr index_t c_offset = CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(
                            c_origin_idx + make_tuple(k, 0, h, w));

                        inner_product<a_vector_t, b_vector_t, FloatC>(
                            a_vec.template AsType<a_vector_t>()[I0],
                            b_vec.template AsType<b_vector_t>()[I0],
                            c_buf(Number<c_offset>{}));
                    });
                });
            });
        });
    }
};

} // namespace ck
#endif
