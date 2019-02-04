#pragma once

template <class Float, class SrcMatrix, class DstMatrix, unsigned NRow, unsigned NCol>
__device__ void
threadwise_matrix_copy(SrcMatrix, Float* const p_src, DstMatrix, Float* p_dst, Sequence<NRow, NCol>)
{
    const auto src_mtx = SrcMatrix{}; // constexpr doesn't compile
    const auto dst_mtx = DstMatrix{}; // constexpr doesn't compile

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
                                Constant<bool, TransA>,
                                FloatA* const p_a_thread,
                                MatrixB,
                                Constant<bool, TransB>,
                                FloatB* const p_b_thread,
                                MatrixC,
                                Constant<bool, TransC>,
                                FloatC* p_c_thread,
                                Accumulator f_accum)
{
    if(TransA && (!TransB) && (!TransC))
    {
        const auto a_mtx = MatrixA{}; // constexpr doesn't compile
        const auto b_mtx = MatrixB{}; // constexpr doesn't compile
        const auto c_mtx = MatrixC{}; // constexpr doesn't compile

        constexpr unsigned M = c_mtx.NRow();
        constexpr unsigned N = c_mtx.NCol();
        constexpr unsigned K = a_mtx.NRow(); // A is transposed

        for(unsigned i = 0; i < M; ++i)
        {
            for(unsigned j = 0; j < N; ++j)
            {
                for(unsigned k = 0; k < K; ++k)
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
