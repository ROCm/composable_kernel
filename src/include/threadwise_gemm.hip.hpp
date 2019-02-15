#pragma once

template <class Float, class SrcMatrix, class DstMatrix, unsigned NRow, unsigned NCol>
__device__ void threadwise_matrix_copy(SrcMatrix,
                                       const Float* __restrict__ p_src,
                                       DstMatrix,
                                       Float* __restrict__ p_dst,
                                       Sequence<NRow, NCol>)
{
    constexpr auto src_mtx = SrcMatrix{};
    constexpr auto dst_mtx = DstMatrix{};

    for(unsigned i = 0; i < NRow; ++i)
    {
        for(unsigned j = 0; j < NCol; ++j)
        {
            const unsigned src_index = src_mtx.Get1dIndex(i, j);
            const unsigned dst_index = dst_mtx.Get1dIndex(i, j);

            p_dst[dst_index] = p_src[src_index];
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

        constexpr unsigned M = c_mtx.NRow();
        constexpr unsigned N = c_mtx.NCol();
        constexpr unsigned K = a_mtx.NRow(); // A is transposed

        for(unsigned k = 0; k < K; ++k)
        {
            for(unsigned i = 0; i < M; ++i)
            {
                for(unsigned j = 0; j < N; ++j)
                {
                    const unsigned aindex = a_mtx.Get1dIndex(k, i); // A is transposed
                    const unsigned bindex = b_mtx.Get1dIndex(k, j);
                    const unsigned cindex = c_mtx.Get1dIndex(i, j);

                    f_accum(p_c_thread[cindex], p_a_thread[aindex] * p_b_thread[bindex]);
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
