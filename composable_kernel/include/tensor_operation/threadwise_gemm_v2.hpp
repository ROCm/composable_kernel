#ifndef CK_THREADWISE_GEMM_V2_HPP
#define CK_THREADWISE_GEMM_V2_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {

// C[M0, M1, N0, N1] += A[K, M0, M1] * B[K, N0, N1]
//   Tensor element can be vectorized data
// Assume:
//   1. ADesc, BDesc, CDesc are known at compile-time
//   2. AOriginIdx, BOriginIdx, COriginIdx are known at compile-time
template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename ADesc,
          typename BDesc,
          typename CDesc,
          typename KLengths,
          typename MLengths,
          typename NLengths,
          typename std::enable_if<ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                                      CDesc::IsKnownAtCompileTime(),
                                  bool>::type = false>
struct ThreadwiseGemm_km0m1_kn0n1_m0m1n0n1
{
    __device__ constexpr ThreadwiseGemm_km0m1_kn0n1_m0m1n0n1()
    {
        static_assert(ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                          CDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        // TODO: sanity-check: compare ADesc, BDesc, CDesc Size with KLenghts, MLengths and NLengths

        // TODO remove this restriction
        static_assert(KLengths::Size() == 1 && MLengths::Size() == 2 && NLengths::Size() == 2,
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

        constexpr auto K  = KLengths{}[I0];
        constexpr auto M0 = MLengths{}[I0];
        constexpr auto M1 = MLengths{}[I1];
        constexpr auto N0 = NLengths{}[I0];
        constexpr auto N1 = NLengths{}[I1];

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        static_for<0, K, 1>{}([&](auto k) {
            static_for<0, M0, 1>{}([&](auto m0) {
                static_for<0, M1, 1>{}([&](auto m1) {
                    static_for<0, N0, 1>{}([&](auto n0) {
                        static_for<0, N1, 1>{}([&](auto n1) {

                            constexpr index_t a_offset =
                                ADesc{}.CalculateOffset(a_origin_idx + make_multi_index(k, m0, m1));
                            constexpr index_t b_offset =
                                BDesc{}.CalculateOffset(b_origin_idx + make_multi_index(k, n0, n1));
                            constexpr index_t c_offset = CDesc{}.CalculateOffset(
                                c_origin_idx + make_multi_index(m0, m1, n0, n1));

                            amd_inner_product_dlop<FloatA, FloatB, FloatC>(
                                a_buf[Number<a_offset>{}],
                                b_buf[Number<b_offset>{}],
                                c_buf(Number<c_offset>{}));
                        });
                    });
                });
            });
        });
    }
};

// C[M0, M1, N0, N1] += A[K0, M0, M1, K1] * B[K0, N0, N1, K1]
//   Tensor element can be vectorized data
// Assume:
//   1. ADesc, BDesc, CDesc are known at compile-time
//   2. AOriginIdx, BOriginIdx, COriginIdx are known at compile-time
template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename ADesc,
          typename BDesc,
          typename CDesc,
          typename KLengths,
          typename MLengths,
          typename NLengths,
          typename std::enable_if<ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                                      CDesc::IsKnownAtCompileTime(),
                                  bool>::type = false>
struct ThreadwiseGemm_k0m0m1k1_k0n0n1k1_m0m1n0n1
{
    __device__ constexpr ThreadwiseGemm_k0m0m1k1_k0n0n1k1_m0m1n0n1()
    {
        static_assert(ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                          CDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        // TODO: sanity-check: compare ADesc, BDesc, CDesc Size with KLenghts, MLengths and NLengths

        // TODO remove this restriction
        static_assert(KLengths::Size() == 2 && MLengths::Size() == 2 && NLengths::Size() == 2,
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

        constexpr index_t K0 = KLengths{}[I0];
        constexpr index_t K1 = KLengths{}[I1];
        constexpr index_t M0 = MLengths{}[I0];
        constexpr index_t M1 = MLengths{}[I1];
        constexpr index_t N0 = NLengths{}[I0];
        constexpr index_t N1 = NLengths{}[I1];

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        static_for<0, K0, 1>{}([&](auto k0) {
            static_for<0, M0, 1>{}([&](auto m0) {
                static_for<0, M1, 1>{}([&](auto m1) {
                    static_for<0, N0, 1>{}([&](auto n0) {
                        static_for<0, N1, 1>{}([&](auto n1) {

                            vector_type<FloatA, K1> a_vec;
                            vector_type<FloatB, K1> b_vec;

                            static_for<0, K1, 1>{}([&](auto k1) {
                                constexpr index_t a_offset = ADesc{}.CalculateOffset(
                                    a_origin_idx + make_multi_index(k0, m0, m1, k1));

                                constexpr index_t b_offset = BDesc{}.CalculateOffset(
                                    b_origin_idx + make_multi_index(k0, n0, n1, k1));

                                a_vec.template AsType<FloatA>()(k1) = a_buf[Number<a_offset>{}];

                                b_vec.template AsType<FloatB>()(k1) = b_buf[Number<b_offset>{}];
                            });

                            using a_vector_t = typename vector_type<FloatA, K1>::type;
                            using b_vector_t = typename vector_type<FloatB, K1>::type;

                            constexpr index_t c_offset = CDesc{}.CalculateOffset(
                                c_origin_idx + make_multi_index(m0, m1, n0, n1));

                            amd_inner_product_dlop<a_vector_t, b_vector_t, FloatC>(
                                a_vec.template AsType<a_vector_t>()[I0],
                                b_vec.template AsType<b_vector_t>()[I0],
                                c_buf(Number<c_offset>{}));
                        });
                    });
                });
            });
        });
    }
};

} // namespace ck
#endif
