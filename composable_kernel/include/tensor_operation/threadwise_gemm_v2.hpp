#ifndef CK_THREADWISE_GEMM_V2_HPP
#define CK_THREADWISE_GEMM_V2_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {

// C[M, N] += transpose(A[K, M]) * B[K, N]
//   Element of matrix can be vectorized data
// Assume:
//   1. ADesc, BDesc, CDesc are known at compile-time
//   2. AOriginIdx, BOriginIdx, COriginIdx are known at compile-time
template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename ADesc,
          typename BDesc,
          typename CDesc,
          typename std::enable_if<ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                                      CDesc::IsKnownAtCompileTime(),
                                  bool>::type = false>
struct ThreadwiseGemm_km_kn_mn_v1r1
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
        static_assert(ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                          CDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(
            is_known_at_compile_time<remove_cv_t<remove_reference_t<AOriginIdx>>>::value &&
                is_known_at_compile_time<remove_cv_t<remove_reference_t<BOriginIdx>>>::value &&
                is_known_at_compile_time<remove_cv_t<remove_reference_t<COriginIdx>>>::value,
            "wrong! AOriginIdx, BOriginIdx, COringinIdx should be known at compile-time");

        static_assert(is_same<remove_cv_t<remove_reference_t<typename ABuffer::type>>,
                              remove_cv_t<remove_reference_t<FloatA>>>::value &&
                      is_same<remove_cv_t<remove_reference_t<typename BBuffer::type>>,
                              remove_cv_t<remove_reference_t<FloatB>>>::value &&
                      is_same<remove_cv_t<remove_reference_t<typename CBuffer::type>>,
                              remove_cv_t<remove_reference_t<FloatC>>>::value &&
                      "wrong! inconsistent type");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto M = CDesc{}.GetLength(I0);
        constexpr auto N = CDesc{}.GetLength(I1);
        constexpr auto K = ADesc{}.GetLength(I0);

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        static_for<0, K, 1>{}([&](auto k) {
            static_for<0, M, 1>{}([&](auto m) {
                constexpr index_t a_offset =
                    ADesc{}.CalculateOffset(a_origin_idx + make_tuple(k, m));

#if 0
                if constexpr(N == 2)
                {
                    constexpr index_t b_offset_0 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(k, I0));
                    constexpr index_t b_offset_1 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(k, I1));

                    constexpr index_t c_offset_0 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(m, I0));
                    constexpr index_t c_offset_1 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(m, I1));

                    amd_assembly_outer_product_1x2(a_buf[Number<a_offset>{}],
                                                   b_buf[Number<b_offset_0>{}],
                                                   b_buf[Number<b_offset_1>{}],
                                                   c_buf(Number<c_offset_0>{}),
                                                   c_buf(Number<c_offset_1>{}));
                }
                else if constexpr(N == 4)
                {
                    constexpr index_t b_offset_0 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(k, I0));
                    constexpr index_t b_offset_1 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(k, I1));
                    constexpr index_t b_offset_2 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(k, I2));
                    constexpr index_t b_offset_3 =
                        BDesc{}.CalculateOffset(b_origin_idx + make_tuple(k, I3));

                    constexpr index_t c_offset_0 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(m, I0));
                    constexpr index_t c_offset_1 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(m, I1));
                    constexpr index_t c_offset_2 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(m, I2));
                    constexpr index_t c_offset_3 =
                        CDesc{}.CalculateOffset(c_origin_idx + make_tuple(m, I3));

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
                    static_for<0, N, 1>{}([&](auto n) {

                        constexpr index_t b_offset =
                            BDesc{}.CalculateOffset(b_origin_idx + make_tuple(k, n));
                        constexpr index_t c_offset =
                            CDesc{}.CalculateOffset(c_origin_idx + make_tuple(m, n));

                        amd_assembly_inner_product(a_buf[Number<a_offset>{}],
                                                   b_buf[Number<b_offset>{}],
                                                   c_buf(Number<c_offset>{}));
                    });
                }
            });
        });
    }
};

} // namespace ck
#endif
