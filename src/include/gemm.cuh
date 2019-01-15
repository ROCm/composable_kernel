#pragma once

template <class ThreadMatrixA,
          class ThreadMatrixB,
          class ThreadMatrixC,
          bool TransA,
          bool TransB,
          bool TransC,
          class FloatA,
          class FloatB,
          class FloatC,
          class Accumulator>
__device__ void threadwise_gemm(ThreadMatrixA,
                                Constant<bool, TransA>,
                                FloatA* const p_a_thread,
                                ThreadMatrixB,
                                Constant<bool, TransB>,
                                FloatB* const p_b_thread,
                                ThreadMatrixC,
                                Constant<bool, TransC>,
                                FloatC* p_c_thread,
                                Accumulator)
{
    // do something
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
          unsigned KPerLoop,
          class Accumulator>
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
        static_assert(ThreadMatrixStrideC > 0, "wrong! ThreadMatrixStrideC == 0!");

#if 0
        constexpr auto a_block_desc = BlockMatrixA{};
        constexpr auto b_block_desc = BlockMatrixB{};

        constexpr unsigned a_thread_row = (!TransA) ? MPerThread : KPerThread;
        constexpr unsigned a_thread_col = (!TransA) ? KPerThread : MPerThread;
        constexpr unsigned b_thread_row = (!TransB) ? KPerThread : NPerThread;
        constexpr unsigned b_thread_col = (!TransB) ? NPerThread : KPerThread;

        constexpr auto a_thread_desc = ConstantMatrixDescriptor<a_thread_row, a_thread_col>{};
        constexpr auto b_thread_desc = ConstantMatrixDescriptor<b_thread_row, b_thread_col>{};
        constexpr auto c_thread_desc = ConstantMatrixDescriptor<MPerThread, NPerThread>{};

        constexpr unsigned m_block = (!TransA) ? a_block_desc.NRow() : a_block_desc.NCol();
        constexpr unsigned n_block = (!TransB) ? b_block_desc.NCol() : b_block_desc.NRow();

        constexpr unsigned m_thread = (!TransA) ? a_thread_desc.NRow() : a_thread_desc.NCol();
        constexpr unsigned n_thread = (!TransB) ? b_thread_desc.NCol() : b_thread_desc.NRow();

        constexpr unsigned num_threads_per_row   = (m_block + m_thread - 1) / m_thread;
        constexpr unsigned num_threads_per_col   = (n_block + n_thread - 1) / n_thread;
        constexpr unsigned num_threads_per_batch = num_threads_per_row * num_threads_per_col;

        static_assert(BlockSize >= ((BatchSize + BatchPerThread - 1) / BatchPerThread) *
                                       num_threads_per_batch,
                      "not enough thread!");

        const auto mtx_c_idnex = CalculateThreadMatrixCIndex(get_thread_local_id());

        // mMyThreadOffsetA = xxx;
        // mMyThreadoffSetB = xxx;
#else
        mMyThreadOffsetA = 0;
        mMyThreadOffsetB = 0;
#endif
    }

    __device__ MatrixIndex CalculateThreadMatrixCIndex(unsigned thread_id) const
    {
#if 0
        constexpr auto a_block = BlockMatrixA{};
        constexpr auto b_block = BlockMatrixB{};
        constexpr auto c_block = BlockMatrixC{};

        constexpr auto a_thread = ThreadMatrixA{};
        constexpr auto b_thread = ThreadMatrixB{};
        constexpr auto c_thread = ThreadMatrixC{};

        constexpr unsigned m_block = (!TransA) ? a_block.NRow() : a_block.NCol();
        constexpr unsigned n_block = (!TransB) ? b_block.NCol() : b_block.NRow();

        constexpr unsigned m_thread = (!TransA) ? a_thread.NRow() : a_thread.NCol();
        constexpr unsigned n_thread = (!TransB) ? b_thread.NCol() : b_thread.NRow();

        constexpr unsigned num_threads_per_row   = (m_block + m_thread - 1) / m_thread;
        constexpr unsigned num_threads_per_col   = (n_block + n_thread - 1) / n_thread;
        constexpr unsigned num_threads_per_batch = num_threads_per_row * num_threads_per_col;

        // this is wrong, need fix
        const unsigned batch_begin = thread_id / (num_threads_per_batch)*BatchPerThread;
        const unsigned tmp = thread_id - batch_id * (num_threads_per_row * num_threads_per_col);
        const unsigned thread_matrix_row_id = tmp / num_threads_per_row;
        const unsigned thread_matrix_col_id = tmp - thread_matrix_row_id * num_threads_per_row;

        return MatrixIndex{
            batch_begin, thread_matrix_row_id * m_thread, thread_matrix_col_id * n_thread};
#else
        return MatrixIndex{0, 0, 0};
#endif
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(FloatA* const p_a_block, FloatB* const p_b_block, FloatC* p_c_thread) const
    {
        // do something
    }
};
