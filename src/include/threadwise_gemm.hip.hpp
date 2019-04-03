#pragma once

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
#elif 1
    static_assert(NCol == 4, "only for NCol == 4");

    using vector_t = typename vector_type<Float, 4>::MemoryType;

    for(index_t i = 0; i < NRow; ++i)
    {
        const index_t src_index = src_mtx.Get1dIndex(i, 0);
        const index_t dst_index = dst_mtx.Get1dIndex(i, 0);

        *(reinterpret_cast<vector_t*>(&p_dst[dst_index])) =
            *(reinterpret_cast<const vector_t*>(&p_src[src_index]));
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
            for(index_t i = 0; i < M; ++i)
            {
                for(index_t j = 0; j < N; ++j)
                {
                    const index_t aindex = a_mtx.Get1dIndex(k, i); // A is transposed
                    const index_t bindex = b_mtx.Get1dIndex(k, j);
                    const index_t cindex = c_mtx.Get1dIndex(i, j);

#if DEVICE_BACKEND_HIP // this only does c += a * b
                    asm volatile("\n \
                                v_mac_f32 %0, %1, %2 \n \
                                "
                                 : "=v"(p_c_thread[cindex])
                                 : "v"(p_a_thread[aindex]),
                                   "v"(p_b_thread[bindex]),
                                   "0"(p_c_thread[cindex]));
#else // this does general accumulation defined by f_accum
                    f_accum(p_c_thread[cindex], p_a_thread[aindex] * p_b_thread[bindex]);
#endif
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
