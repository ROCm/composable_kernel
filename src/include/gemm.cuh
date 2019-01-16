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

template <unsigned BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class ThreadMatrixC,
          bool TransA,
          bool TransB,
          bool TransC,
          unsigned BlockMatrixStrideA,
          unsigned BlockMatrixStrideB,
          unsigned ThreadMatrixStrideC,
          unsigned BatchSize,
          unsigned BatchPerThread,
          unsigned KPerThreadLoop,
          bool DistributeThreadAlongColumnFirst>
struct blockwise_1d_strided_batched_gemm_block_a_block_b_thread_c
{
    unsigned mMyThreadOffsetA = 0;
    unsigned mMyThreadOffsetB = 0;

    struct MatrixIndex
    {
        unsigned batch_begin;
        unsigned row_begin;
        unsigned col_begin;
    };

    __device__ blockwise_1d_strided_batched_gemm_block_a_block_b_thread_c()
    {
        const auto a_block_mtx = BlockMatrixA{}; // constexpr doesn't compile
        const auto b_block_mtx = BlockMatrixB{}; // constexpr doesn't compile

        const auto c_thread_mtx_index = CalculateThreadMatrixCIndex(get_thread_local_1d_id());

        mMyThreadOffsetA = c_thread_mtx_index.batch_begin * a_block_mtx.GetElementSpace() +
                           ((!TransA) ? a_block_mtx.Get1dIndex(c_thread_mtx_index.row_begin, 0)
                                      : a_block_mtx.Get1dIndex(0, c_thread_mtx_index.row_begin));

        mMyThreadOffsetB = c_thread_mtx_index.batch_begin * b_block_mtx.GetElementSpace() +
                           ((!TransB) ? b_block_mtx.Get1dIndex(0, c_thread_mtx_index.col_begin)
                                      : b_block_mtx.Get1dIndex(c_thread_mtx_index.col_begin, 0));
    }

    __device__ MatrixIndex CalculateThreadMatrixCIndex(unsigned thread_id) const
    {

        if(TransA && (!TransB) && (!TransC))
        {
            const auto a_block_mtx = BlockMatrixA{}; // constexpr doesn't compile
            const auto b_block_mtx = BlockMatrixB{}; // constexpr doesn't compile

            static_assert(a_block_mtx.NRow() == b_block_mtx.NRow(),
                          "wrong! k dimension not consistent!");

            constexpr unsigned MPerBlock = a_block_mtx.NCol();
            constexpr unsigned NPerBlock = b_block_mtx.NCol();

            const auto c_thread_mtx = ThreadMatrixC{}; // constexpr doesn't compile

            // divide thread work
            constexpr unsigned MPerThread = c_thread_mtx.NRow();
            constexpr unsigned NPerThread = c_thread_mtx.NCol();

            static_assert(BatchSize % BatchPerThread == 0, "BatchSize % BatchPerThread != 0");
            static_assert(MPerBlock % MPerThread == 0, "MPerBlock % MPerThread != 0");
            static_assert(NPerBlock % NPerThread == 0, "NPerBlock % NPerThread != 0");

            constexpr unsigned BThreadWork = (BatchSize + BatchPerThread - 1) / BatchPerThread;
            constexpr unsigned MThreadWork = (MPerBlock + MPerThread - 1) / MPerThread;
            constexpr unsigned NThreadWork = (NPerBlock + NPerThread - 1) / NPerThread;

            static_assert(BlockSize == BThreadWork * MThreadWork * NThreadWork,
                          "wrong! wrong BlockSize");

            if(DistributeThreadAlongColumnFirst)
            {
                // num of operations can be reduced
                const unsigned b_work_id = thread_id / (MThreadWork * NThreadWork);
                unsigned itmp            = thread_id - b_work_id * (MThreadWork * NThreadWork);
                const unsigned m_work_id = itmp / NThreadWork;
                const unsigned n_work_id = itmp - m_work_id * NThreadWork;

                return MatrixIndex{
                    b_work_id * BatchPerThread, m_work_id * MPerThread, n_work_id * NPerThread};
            }
            else
            {
                // not implemented
                assert(false);
            }
        }
        else
        {
            // not implemented
            assert(false);
        }
    }

    template <class FloatA, class FloatB, class FloatC, class Accumulator>
    __device__ void run(FloatA* const p_a_block,
                        FloatB* const p_b_block,
                        FloatC* p_c_thread,
                        Accumulator f_accum) const
    {
        if(TransA && (!TransB) && (!TransC))
        {
            constexpr auto True  = Constant<bool, true>{};
            constexpr auto False = Constant<bool, false>{};

            const auto a_block_mtx  = BlockMatrixA{};  // constexpr doesn't compile
            const auto b_block_mtx  = BlockMatrixB{};  // constexpr doesn't compile
            const auto c_thread_mtx = ThreadMatrixC{}; // constexpr doesn't compile

            constexpr unsigned KPerBlock = a_block_mtx.NRow(); // A is transposed

            constexpr unsigned MPerThread = c_thread_mtx.NRow();
            constexpr unsigned NPerThread = c_thread_mtx.NCol();

            // a is transposed, b is not
            const auto a_thread_mtx = make_ConstantMatrixDescriptor(
                Number<KPerThreadLoop>{}, Number<MPerThread>{}); // constexpr doesn't compile

            const auto b_thread_mtx = make_ConstantMatrixDescriptor(
                Number<KPerThreadLoop>{}, Number<NPerThread>{}); // constexpr doesn't compile

            FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
            FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

            // loop over k
            for(unsigned k_begin = 0; k_begin < KPerBlock; k_begin += KPerThreadLoop)
            {
                // read first batch of a, b
                threadwise_matrix_copy(a_block_mtx,
                                       p_a_block + mMyThreadOffsetA +
                                           k_begin * a_block_mtx.RowStride(),
                                       a_thread_mtx,
                                       p_a_thread,
                                       a_thread_mtx.GetLengths());

                threadwise_matrix_copy(b_block_mtx,
                                       p_b_block + mMyThreadOffsetB +
                                           k_begin * b_block_mtx.RowStride(),
                                       b_thread_mtx,
                                       p_b_thread,
                                       b_thread_mtx.GetLengths());

                // loop over batch
                for(unsigned ib = 0; ib + 1 < BatchPerThread; ++ib)
                {
                    // do current batch of gemm
                    threadwise_gemm(a_thread_mtx,
                                    True,
                                    p_a_thread,
                                    b_thread_mtx,
                                    False,
                                    p_b_thread,
                                    c_thread_mtx,
                                    False,
                                    p_c_thread + ib * ThreadMatrixStrideC,
                                    f_accum);

                    // read next batch of a, b
                    if(BlockMatrixStrideA != 0)
                    {
                        threadwise_matrix_copy(a_block_mtx,
                                               p_a_block + mMyThreadOffsetA +
                                                   (ib + 1) * BlockMatrixStrideA +
                                                   +k_begin * a_block_mtx.RowStride(),
                                               a_thread_mtx,
                                               p_a_thread,
                                               a_thread_mtx.GetLengths());
                    }

                    if(BlockMatrixStrideB != 0)
                    {
                        threadwise_matrix_copy(b_block_mtx,
                                               p_b_block + mMyThreadOffsetB +
                                                   (ib + 1) * BlockMatrixStrideB +
                                                   k_begin * b_block_mtx.RowStride(),
                                               b_thread_mtx,
                                               p_b_thread,
                                               b_thread_mtx.GetLengths());
                    }
                }

                // do last batch of gemm
                threadwise_gemm(a_thread_mtx,
                                True,
                                p_a_thread,
                                b_thread_mtx,
                                False,
                                p_b_thread,
                                c_thread_mtx,
                                False,
                                p_c_thread + (BatchPerThread - 1) * ThreadMatrixStrideC,
                                f_accum);
            }
        }
    }
};
