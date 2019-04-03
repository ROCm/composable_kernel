#pragma once

#include "inline_asm.hpp"

template <class Float, class SrcMatrix, class DstMatrix, index_t NRow, index_t NCol>
__device__ void threadwise_matrix_copy(SrcMatrix,
                                       const Float* __restrict__ p_src,
                                       DstMatrix,
                                       Float* __restrict__ p_dst,
                                       Sequence<NRow, NCol>)
{
    constexpr auto src_mtx = SrcMatrix{};
    constexpr auto dst_mtx = DstMatrix{};

#if 1
    for(index_t i = 0; i < NRow; ++i)
    {
        for(index_t j = 0; j < NCol; ++j)
        {
            const index_t src_index = src_mtx.Get1dIndex(i, j);
            const index_t dst_index = dst_mtx.Get1dIndex(i, j);

            p_dst[dst_index] = p_src[src_index];
        }
    }
#else
    static_assert(NCol == 4, "only for NCol == 4");

    for(index_t i = 0; i < NRow; ++i)
    {
        const index_t src_index = src_mtx.Get1dIndex(i, 0);
        const index_t dst_index = dst_mtx.Get1dIndex(i, 0);

        Float4 *reg_p = (Float4 *)&p_dst[dst_index];
        Float4 *loc_p = (Float4 *)&p_src[src_index];

        ds_read_b128(reg_p[0], (void *)&loc_p[0]);
    }
#endif
}

template <class MatrixA,
          class MatrixB,
          class MatrixC,
          bool TransA,
          bool TransB,
          bool TransC,
          class FloatA,
          class FloatB,
          class FloatC,
          class Accumulator>
__device__ void threadwise_gemm(MatrixA,
                                integral_constant<bool, TransA>,
                                const FloatA* __restrict__ p_a_thread,
                                MatrixB,
                                integral_constant<bool, TransB>,
                                const FloatB* __restrict__ p_b_thread,
                                MatrixC,
                                integral_constant<bool, TransC>,
                                FloatC* __restrict__ p_c_thread,
                                Accumulator f_accum)
{
    if(TransA && (!TransB) && (!TransC))
    {
        constexpr auto a_mtx = MatrixA{};
        constexpr auto b_mtx = MatrixB{};
        constexpr auto c_mtx = MatrixC{};

        constexpr index_t M = c_mtx.NRow();
        constexpr index_t N = c_mtx.NCol();
        constexpr index_t K = a_mtx.NRow(); // A is transposed

        for(index_t k = 0; k < K; ++k)
        {
#if 1
            for(index_t i = 0; i < M; i+=4)
            {
                const index_t aindex = a_mtx.Get1dIndex(k, i); // A is transposed
                const Float4 *a_vec = (const Float4 *)&p_a_thread[aindex];

                for(index_t j = 0; j < N; j+=4)
                {
                    const index_t bindex = b_mtx.Get1dIndex(k, j);
                    const index_t cindex = c_mtx.Get1dIndex(i, j);

                    const Float4 *b_vec = (const Float4 *)&p_b_thread[bindex];
                    Float4 *c_vec = (Float4 *)&p_c_thread[cindex];

                    outerProduct4x4(a_vec[0], b_vec[0], c_vec[0], c_vec[2], c_vec[4], c_vec[6]);
                }
            }
#else
            const Float4 *a_vec = (const Float4 *)p_a_thread;
            const Float4 *b_vec = (const Float4 *)p_b_thread;
            Float4 *c_vec = (Float4 *)p_c_thread;

            outerProduct8x8(a_vec, b_vec, c_vec);
#endif
        }
    }
    else
    {
        // not implemented
        assert(false);
    }
}
