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

    for(index_t i = 0; i < NRow; ++i)
    {
        // optimize for vector-4 load
        if(NCol % 4 == 0)
        {
            using vector_t = typename vector_type<Float, 4>::MemoryType;

            for(index_t j = 0; j < NCol / 4; ++j)
            {
                const index_t src_index = src_mtx.Get1dIndex(i, 4 * j);
                const index_t dst_index = dst_mtx.Get1dIndex(i, 4 * j);

                *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
                    *reinterpret_cast<const vector_t*>(&p_src[src_index]);
            }
        }
        else
        {
            for(index_t j = 0; j < NCol; ++j)
            {
                const index_t src_index = src_mtx.Get1dIndex(i, j);
                const index_t dst_index = dst_mtx.Get1dIndex(i, j);

                p_dst[dst_index] = p_src[src_index];
            }
        }
    }
}

template <class MatrixA,
          class MatrixB,
          class MatrixC,
          bool TransA,
          bool TransB,
          bool TransC,
          class FloatA,
          class FloatB,
          class FloatC>
__device__ void threadwise_gemm(MatrixA,
                                integral_constant<bool, TransA>,
                                const FloatA* __restrict__ p_a_thread,
                                MatrixB,
                                integral_constant<bool, TransB>,
                                const FloatB* __restrict__ p_b_thread,
                                MatrixC,
                                integral_constant<bool, TransC>,
                                FloatC* __restrict__ p_c_thread)
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
            for(index_t i = 0; i < M; i++)
            {
                for(index_t j = 0; j < N; j++)
                {
                    const index_t aindex = a_mtx.Get1dIndex(k, i); // A is transposed
                    const index_t bindex = b_mtx.Get1dIndex(k, j);
                    const index_t cindex = c_mtx.Get1dIndex(i, j);

                    p_c_thread[cindex] += p_a_thread[aindex] * p_b_thread[bindex];
                }
            }
        }
    }
    else
    {
        // not implemented
        assert(false);
    }
}
