#pragma once
#include "common.hpp"
#include "ConstantMatrixDescriptor.hpp"

template <class Float, class Matrix>
__device__ void threadwise_matrix_set_zero(Matrix, Float* __restrict__ p_thread)
{
    for(index_t i = 0; i < Matrix::NRow(); ++i)
    {
        for(index_t j = 0; j < Matrix::NCol(); ++j)
        {
            const index_t id = Matrix::GetOffsetFromMultiIndex(i, j);
            p_thread[id]     = Float(0);
        }
    }
}

template <class Float,
          class SrcMatrix,
          class DstMatrix,
          index_t NRow,
          index_t NCol,
          index_t DataPerRead>
__device__ void threadwise_matrix_copy(SrcMatrix,
                                       const Float* __restrict__ p_src,
                                       DstMatrix,
                                       Float* __restrict__ p_dst,
                                       Sequence<NRow, NCol>,
                                       Number<DataPerRead>)
{
    static_assert(NCol % DataPerRead == 0, "wrong! should be NCol % == DataPerRead == 0");

    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    constexpr auto src_mtx = SrcMatrix{};
    constexpr auto dst_mtx = DstMatrix{};

    for(index_t i = 0; i < NRow; ++i)
    {
        for(index_t j = 0; j < NCol; j += DataPerRead)
        {
            const index_t src_index = src_mtx.GetOffsetFromMultiIndex(i, j);
            const index_t dst_index = dst_mtx.GetOffsetFromMultiIndex(i, j);

            *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
                *reinterpret_cast<const vector_t*>(&p_src[src_index]);
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
#if 0
    if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
    {
        printf("p_a_thread: %f %f %f %f\n",
               p_a_thread[0],
               p_a_thread[1],
               p_a_thread[2],
               p_a_thread[3]);
        printf("p_b_thread: %f %f %f %f\n",
               p_b_thread[0],
               p_b_thread[1],
               p_b_thread[2],
               p_b_thread[3]);
    }
#endif

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
            for(index_t i = 0; i < M; ++i)
            {
                for(index_t j = 0; j < N; ++j)
                {
                    const index_t aindex = a_mtx.GetOffsetFromMultiIndex(k, i); // A is transposed
                    const index_t bindex = b_mtx.GetOffsetFromMultiIndex(k, j);
                    const index_t cindex = c_mtx.GetOffsetFromMultiIndex(i, j);

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
