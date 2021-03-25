#ifndef CK_THREADWISE_GEMM_V2_HPP
#define CK_THREADWISE_GEMM_V2_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {

template <typename Float, typename Desc>
__device__ void threadwise_matrix_set_zero_v2(Desc, Float* __restrict__ p_thread)
{
    static_assert(Desc::IsKnownAtCompileTime(), "wrong! Desc should be known at compile-time");

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    constexpr auto desc = Desc{};

    constexpr auto M = desc.GetLength(I0);
    constexpr auto N = desc.GetLength(I1);

    static_for<0, M, 1>{}([&](auto i) {
        static_for<0, N, 1>{}([&](auto j) {
            constexpr auto offset = desc.CalculateOffset(make_tuple(i, j));

            p_thread[offset] = Float(0);
        });
    });
}

template <typename SrcDesc,
          typename DstDesc,
          index_t NSliceRow,
          index_t NSliceCol,
          index_t DataPerAccess>
struct ThreadwiseMatrixSliceCopy_v2
{
    template <typename Data>
    __device__ static void Run(const Data* p_src, Data* p_dst)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        using vector_t = typename vector_type<Data, DataPerAccess>::type;

        static_for<0, NSliceRow, 1>{}([&](auto i) {
            static_for<0, NSliceCol, DataPerAccess>{}([&](auto j) {
                constexpr auto src_offset = SrcDesc{}.CalculateOffset(make_tuple(i, j));
                constexpr auto dst_offset = DstDesc{}.CalculateOffset(make_tuple(i, j));

                *reinterpret_cast<vector_t*>(&p_dst[dst_offset]) =
                    *reinterpret_cast<const vector_t*>(&p_src[src_offset]);
            });
        });
    }
};

// C[M, N] += transpose(A[K, M]) * B[K, N]
//   Element of matrix can be vectorized data
template <typename ADesc,
          typename BDesc,
          typename CDesc,
          typename std::enable_if<ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                                      CDesc::IsKnownAtCompileTime(),
                                  bool>::type = false>
struct ThreadwiseGemm_km_kn_mn_v1
{
    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run_source(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        static_assert(ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                          CDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto M = CDesc{}.GetLength(I0);
        constexpr auto N = CDesc{}.GetLength(I1);
        constexpr auto K = ADesc{}.GetLength(I0);

        static_for<0, K, 1>{}([&](auto k) {
            static_for<0, M, 1>{}([&](auto m) {
                static_for<0, N, 1>{}([&](auto n) {
                    constexpr auto a_offset = ADesc{}.CalculateOffset(make_tuple(k, m));
                    constexpr auto b_offset = BDesc{}.CalculateOffset(make_tuple(k, n));
                    constexpr auto c_offset = CDesc{}.CalculateOffset(make_tuple(m, n));

                    p_c[c_offset] +=
                        inner_product_with_conversion<FloatC>{}(p_a[a_offset], p_b[b_offset]);
                });
            });
        });
    }

#if CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM
    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run_amd_asm(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        static_assert(ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                          CDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto M = CDesc{}.GetLength(I0);
        constexpr auto N = CDesc{}.GetLength(I1);
        constexpr auto K = ADesc{}.GetLength(I0);

        static_assert(N == 4 || N == 2, "wrong! this config not supported by asm yet");

        static_for<0, K, 1>{}([&](auto k) {
            static_for<0, M, 1>{}([&](auto m) {
                constexpr auto a_offset = ADesc{}.CalculateOffset(make_tuple(k, m));

                if constexpr(N == 2)
                {
                    constexpr auto b_offset_0 = BDesc{}.CalculateOffset(make_tuple(k, I0));
                    constexpr auto b_offset_1 = BDesc{}.CalculateOffset(make_tuple(k, I1));

                    constexpr auto c_offset_0 = CDesc{}.CalculateOffset(make_tuple(m, I0));
                    constexpr auto c_offset_1 = CDesc{}.CalculateOffset(make_tuple(m, I1));

                    amd_assembly_outer_product_1x2(p_a[a_offset],
                                                   p_b[b_offset_0],
                                                   p_b[b_offset_1],
                                                   p_c[c_offset_0],
                                                   p_c[c_offset_1]);
                }
                else if constexpr(N == 4)
                {
                    constexpr auto b_offset_0 = BDesc{}.CalculateOffset(make_tuple(k, I0));
                    constexpr auto b_offset_1 = BDesc{}.CalculateOffset(make_tuple(k, I1));
                    constexpr auto b_offset_2 = BDesc{}.CalculateOffset(make_tuple(k, I2));
                    constexpr auto b_offset_3 = BDesc{}.CalculateOffset(make_tuple(k, I3));

                    constexpr auto c_offset_0 = CDesc{}.CalculateOffset(make_tuple(m, I0));
                    constexpr auto c_offset_1 = CDesc{}.CalculateOffset(make_tuple(m, I1));
                    constexpr auto c_offset_2 = CDesc{}.CalculateOffset(make_tuple(m, I2));
                    constexpr auto c_offset_3 = CDesc{}.CalculateOffset(make_tuple(m, I3));

                    amd_assembly_outer_product_1x4(p_a[a_offset],
                                                   p_b[b_offset_0],
                                                   p_b[b_offset_1],
                                                   p_b[b_offset_2],
                                                   p_b[b_offset_3],
                                                   p_c[c_offset_0],
                                                   p_c[c_offset_1],
                                                   p_c[c_offset_2],
                                                   p_c[c_offset_3]);
                }
            });
        });
    }
#endif

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
#if CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM
        Run_amd_asm(p_a, p_b, p_c);
#else
        Run_source(p_a, p_b, p_c);
#endif
    }
};

} // namespace ck
#endif
