#pragma once

template <class ThreadMatrixA,
          bool TransA,
          class FloatA,
          class ThreadMatrixB,
          bool TransB,
          class FloatB,
          class ThreadMatrixC,
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
          bool TransA,
          bool TransB,
          unsigned BatchSize,
          unsigned BlockMatrixStrideA,
          unsigned BlockMatrixStrideB,
          unsigned BatchPerThread,
          unsigned MPerThread,
          unsigned NPerThread,
          unsigned KPerThread,
          class Accumulator>
struct blockwise_1d_strided_batched_gemm_block_a_block_b_thread_c
{
    struct MatrixIndex
    {
        unsigned batch_begin;
        unsigned block_row_begin;
        unsigned block_col_begin;
    };

    __device__ blockwise_1d_strided_batched_gemm_block_a_block_b_thread_c()
    {
        static_assert(ThreadMatrixStrideC > 0, "wrong! ThreadMatrixStrideC == 0!");

        constexpr auto a_block = BlockMatrixA{};
        constexpr auto b_block = BlockMatrixB{};

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

        static_assert(BlockSize >= ((BatchSize + BatchPerThread - 1) / BatchPerThread) *
                                       num_threads_per_batch,
                      "not enough thread!");

        const auto mtx_c_idnex = CalculateThreadMatrixCIndex(get_thread_local_id());

        mMyThreadOffsetA = xxx;
        mMyThreadoffSetB = xxx;
    }

    __device__ MatrixIndex CalculateThreadMatrixCIndex(unsigned thread_id) const
    {
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
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void run(FloatA* const p_a_block, FloatB* const p_b_block, FloatC* p_c_thread) const
    {
        // do something
    }

    private:
    unsigned mMyThreadOffsetA = 0;
    unsigned mMyThreadOffsetB = 0;
}
