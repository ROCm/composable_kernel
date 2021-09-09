#ifndef CK_THREADWISE_GEMM_DLOPS_V3_HPP
#define CK_THREADWISE_GEMM_DLOPS_V3_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {

// C[M, N] += transpose(A[K, M]) * B[K, N]
//   Element of matrix can be vectorized data
// Assume:
//   1. AThreadDesc_E_K, BThreadDesc_E_N_Ho_Wo, CThreadDesc_K_N_Ho_Wo are known at compile-time
//   2. AOriginIdx, BOriginIdx, COriginIdx are known at compile-time
template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AThreadDesc_E_K,
          typename BThreadDesc_E_N_Ho_Wo,
          typename CThreadDesc_K_N_Ho_Wo,
          index_t H,
          index_t W,
          typename enable_if<AThreadDesc_E_K::IsKnownAtCompileTime() &&
                                 BThreadDesc_E_N_Ho_Wo::IsKnownAtCompileTime() &&
                                 CThreadDesc_K_N_Ho_Wo::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseGemmDlops_km_kn_mn_v3
{

    __device__ ThreadwiseGemmDlops_km_kn_mn_v3()
    {
        static_assert(AThreadDesc_E_K::IsKnownAtCompileTime() &&
                          BThreadDesc_E_N_Ho_Wo::IsKnownAtCompileTime() &&
                          CThreadDesc_K_N_Ho_Wo::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");
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

        static_assert(AThreadDesc_E_K::IsKnownAtCompileTime() &&
                          BThreadDesc_E_N_Ho_Wo::IsKnownAtCompileTime() &&
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

        constexpr auto E = AThreadDesc_E_K{}.GetLength(I0);
        constexpr auto K = AThreadDesc_E_K{}.GetLength(I1);

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        static_for<0, E, 1>{}([&](auto e) {
            static_for<0, K, 1>{}([&](auto k) {
                constexpr index_t a_offset =
                    AThreadDesc_E_K{}.CalculateOffset(a_origin_idx + make_tuple(e, k));

#if 0
                if constexpr(H == 2 && W == 2)
                {
                    constexpr index_t b_offset_0 =
                        BThreadDesc_E_N_Ho_Wo{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 0, 0));
                    constexpr index_t b_offset_1 =
                        BThreadDesc_E_N_Ho_Wo{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 0, 1));
                    constexpr index_t b_offset_2 =
                        BThreadDesc_E_N_Ho_Wo{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 1, 0));
                    constexpr index_t b_offset_3 =
                        BThreadDesc_E_N_Ho_Wo{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 1, 1));

                    constexpr index_t c_offset_0 =
                        CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 0, 0));
                    constexpr index_t c_offset_1 =
                        CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 0, 1));
                    constexpr index_t c_offset_2 =
                        CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 1, 0));
                    constexpr index_t c_offset_3 =
                        CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 1, 1));

                    amd_assembly_outer_product_1x4(a_buf[Number<a_offset>{}],
                                                   b_buf[Number<b_offset_0>{}],
                                                   b_buf[Number<b_offset_1>{}],
                                                   b_buf[Number<b_offset_2>{}],
                                                   b_buf[Number<b_offset_3>{}],
                                                   c_buf(Number<c_offset_0>{}),
                                                   c_buf(Number<c_offset_1>{}),
                                                   c_buf(Number<c_offset_2>{}),
                                                   c_buf(Number<c_offset_3>{}));
                }
                else if constexpr(H == 4 && W == 1)
                {
                    constexpr index_t b_offset_0 =
                        BThreadDesc_E_N_Ho_Wo{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 0, 0));
                    constexpr index_t b_offset_1 =
                        BThreadDesc_E_N_Ho_Wo{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 1, 0));
                    constexpr index_t b_offset_2 =
                        BThreadDesc_E_N_Ho_Wo{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 2, 0));
                    constexpr index_t b_offset_3 =
                        BThreadDesc_E_N_Ho_Wo{}.CalculateOffset(b_origin_idx + make_tuple(e, 0, 3, 0));

                    constexpr index_t c_offset_0 =
                        CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 0, 0));
                    constexpr index_t c_offset_1 =
                        CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 1, 0));
                    constexpr index_t c_offset_2 =
                        CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 2, 0));
                    constexpr index_t c_offset_3 =
                        CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx + make_tuple(k, 0, 3, 0));

                    amd_assembly_outer_product_1x4(a_buf[Number<a_offset>{}],
                                                   b_buf[Number<b_offset_0>{}],
                                                   b_buf[Number<b_offset_1>{}],
                                                   b_buf[Number<b_offset_2>{}],
                                                   b_buf[Number<b_offset_3>{}],
                                                   c_buf(Number<c_offset_0>{}),
                                                   c_buf(Number<c_offset_1>{}),
                                                   c_buf(Number<c_offset_2>{}),
                                                   c_buf(Number<c_offset_3>{}));
                }
                else
#endif
                {
                    static_for<0, H, 1>{}([&](auto h) {
                        static_for<0, W, 1>{}([&](auto w) {
                            constexpr index_t b_offset = BThreadDesc_E_N_Ho_Wo{}.CalculateOffset(
                                b_origin_idx + make_tuple(e, 0, h, w));

                            constexpr index_t c_offset = CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(
                                c_origin_idx + make_tuple(k, 0, h, w));

                            c_buf(Number<c_offset>{}) += inner_product_with_conversion<FloatC>{}(
                                a_buf[Number<a_offset>{}], b_buf[Number<b_offset>{}]);
                        });
                    });
                }
            });
        });
    }
};

} // namespace ck
#endif
