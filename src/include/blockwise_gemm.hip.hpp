#pragma once
#include "threadwise_gemm.hip.hpp"

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
struct Blockwise1dStridedBatchedGemmBlockABlockBThreadC
{
    unsigned mMyThreadOffsetA = 0;
    unsigned mMyThreadOffsetB = 0;

    struct MatrixIndex
    {
        unsigned batch;
        unsigned row;
        unsigned col;
    };

    __device__ Blockwise1dStridedBatchedGemmBlockABlockBThreadC()
    {
        constexpr auto a_block_mtx = BlockMatrixA{};
        constexpr auto b_block_mtx = BlockMatrixB{};

        const auto c_thread_mtx_index = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        mMyThreadOffsetA = c_thread_mtx_index.batch * BlockMatrixStrideA +
                           ((!TransA) ? a_block_mtx.Get1dIndex(c_thread_mtx_index.row, 0)
                                      : a_block_mtx.Get1dIndex(0, c_thread_mtx_index.row));

        mMyThreadOffsetB = c_thread_mtx_index.batch * BlockMatrixStrideB +
                           ((!TransB) ? b_block_mtx.Get1dIndex(0, c_thread_mtx_index.col)
                                      : b_block_mtx.Get1dIndex(c_thread_mtx_index.col, 0));

#if 0
        if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
        {
            print_ConstantMatrixDescriptor(BlockMatrixA{}, "a_block_mtx: ");
            print_ConstantMatrixDescriptor(BlockMatrixB{}, "b_block_mtx: ");
            print_ConstantMatrixDescriptor(ThreadMatrixC{}, "c_thread_mtx: ");

            printf("%u %u, %u %u %u, %u %u\n",
                   get_block_1d_id(),
                   get_thread_local_1d_id(),
                   c_thread_mtx_index.batch,
                   c_thread_mtx_index.row,
                   c_thread_mtx_index.col,
                   mMyThreadOffsetA,
                   mMyThreadOffsetB);
        }
#endif
    }

    __device__ MatrixIndex GetBeginOfThreadMatrixC(unsigned thread_id) const
    {

        if(TransA && (!TransB) && (!TransC))
        {
            constexpr auto a_block_mtx = BlockMatrixA{};
            constexpr auto b_block_mtx = BlockMatrixB{};

            static_assert(a_block_mtx.NRow() == b_block_mtx.NRow(),
                          "wrong! k dimension not consistent!");

            constexpr unsigned MPerBlock = a_block_mtx.NCol();
            constexpr unsigned NPerBlock = b_block_mtx.NCol();

            constexpr auto c_thread_mtx = ThreadMatrixC{};

            // divide thread work
            constexpr unsigned MPerThread = c_thread_mtx.NRow();
            constexpr unsigned NPerThread = c_thread_mtx.NCol();

            static_assert(BatchSize % BatchPerThread == 0, "BatchSize % BatchPerThread != 0");
            static_assert(MPerBlock % MPerThread == 0, "MPerBlock % MPerThread != 0");
            static_assert(NPerBlock % NPerThread == 0, "NPerBlock % NPerThread != 0");

            constexpr unsigned BatchThreadWork = (BatchSize + BatchPerThread - 1) / BatchPerThread;
            constexpr unsigned MThreadWork     = (MPerBlock + MPerThread - 1) / MPerThread;
            constexpr unsigned NThreadWork     = (NPerBlock + NPerThread - 1) / NPerThread;

            static_assert(BlockSize == BatchThreadWork * MThreadWork * NThreadWork,
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

    // this should be optimized away if input is known
    __device__ static MatrixIndex
    GetDistanceFromBeginOfThreadMatrixC(unsigned batch_in_c, unsigned m_in_c, unsigned n_in_c)
    {
        return MatrixIndex{batch_in_c, m_in_c, n_in_c};
    }

    template <class FloatA, class FloatB, class FloatC, class Accumulator>
    __device__ void Run(const FloatA* __restrict__ p_a_block,
                        const FloatB* __restrict__ p_b_block,
                        FloatC* __restrict__ p_c_thread,
                        Accumulator f_accum) const
    {
        if(TransA && (!TransB) && (!TransC))
        {
            constexpr auto True  = integral_constant<bool, true>{};
            constexpr auto False = integral_constant<bool, false>{};

            constexpr auto a_block_mtx  = BlockMatrixA{};
            constexpr auto b_block_mtx  = BlockMatrixB{};
            constexpr auto c_thread_mtx = ThreadMatrixC{};

            constexpr unsigned KPerBlock = a_block_mtx.NRow(); // A is transposed

            constexpr unsigned MPerThread = c_thread_mtx.NRow();
            constexpr unsigned NPerThread = c_thread_mtx.NCol();

            // a is transposed, b is not
            constexpr auto a_thread_mtx =
                make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

            constexpr auto b_thread_mtx =
                make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

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

template <unsigned BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class ThreadMatrixC,
          unsigned BlockMatrixStrideA,
          unsigned BlockMatrixStrideB,
          unsigned ThreadMatrixStrideC,
          unsigned BatchSize,
          unsigned MPerThreadSubC,
          unsigned NPerThreadSubC,
          unsigned MLevel0Cluster,
          unsigned NLevel0Cluster,
          unsigned MLevel1Cluster,
          unsigned NLevel1Cluster,
          unsigned KPerThreadLoop,
          unsigned BatchPerThread>
struct BlockwiseBatchGemmBlockABlockBThreadCTransANormalBNormalC_V2
{
    unsigned mMyThreadOffsetA = 0;
    unsigned mMyThreadOffsetB = 0;

    struct MatrixIndex
    {
        unsigned batch;
        unsigned row;
        unsigned col;
    };

    __device__ BlockwiseBatchGemmBlockABlockBThreadCTransANormalBNormalC_V2()
    {
        static_assert(BatchSize % BatchPerThread == 0,
                      "wrong! BatchSize is not dividable by BatchPerThread");

        constexpr unsigned BatchThreadWork = BatchSize / BatchPerThread;

        constexpr unsigned ThreadPerLevel1Cluster =
            MLevel0Cluster * NLevel0Cluster * MLevel1Cluster * NLevel1Cluster;

        static_assert(BlockSize == BatchThreadWork * ThreadPerLevel1Cluster,
                      "wrong! wrong blocksize\n");

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        static_assert(a_block_mtx.NRow() == b_block_mtx.NRow(),
                      "wrong! K dimension not consistent\n");

        constexpr unsigned M = a_block_mtx.NCol(); // A is transposed
        constexpr unsigned N = b_block_mtx.NCol();
        constexpr unsigned K = a_block_mtx.NRow();

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        static_assert((MPerThread % MPerThreadSubC == 0) && (NPerThread % NPerThreadSubC == 0),
                      "wrong! Cannot evenly divide thread work among repeat \n");

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

        static_assert((M % MRepeat == 0) && (N % NRepeat == 0),
                      "wrong! Cannot evenly divide work among repeat\n");

        constexpr unsigned MPerLevel1Cluster = M / MRepeat;
        constexpr unsigned NPerLevel1Cluster = N / NRepeat;

        static_assert((MPerLevel1Cluster % MLevel1Cluster == 0) &&
                          (NPerLevel1Cluster % NLevel1Cluster == 0),
                      "wrong! Cannot evenly divide work among Level1Cluster\n");

        constexpr unsigned MPerLevel0Cluster = MPerLevel1Cluster / MLevel1Cluster;
        constexpr unsigned NPerLevel0Cluster = NPerLevel1Cluster / NLevel1Cluster;

        static_assert((MPerLevel0Cluster % MLevel0Cluster == 0) &&
                          (NPerLevel0Cluster % NLevel0Cluster == 0),
                      "wrong! Cannot evenly divide work among Level0Cluster\n");

        static_assert((MPerThreadSubC == MPerLevel0Cluster / MLevel0Cluster) &&
                          (NPerThreadSubC == NPerLevel0Cluster / NLevel0Cluster),
                      "wrong! thread work size is wrong\n");

        const auto c_thread_mtx_index = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        mMyThreadOffsetA = c_thread_mtx_index.batch * BlockMatrixStrideA +
                           a_block_mtx.Get1dIndex(0, c_thread_mtx_index.row);

        mMyThreadOffsetB = c_thread_mtx_index.batch * BlockMatrixStrideB +
                           b_block_mtx.Get1dIndex(0, c_thread_mtx_index.col);

#if 0
        if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
        {
            print_ConstantMatrixDescriptor(BlockMatrixA{}, "a_block_mtx: ");
            print_ConstantMatrixDescriptor(BlockMatrixB{}, "b_block_mtx: ");
            print_ConstantMatrixDescriptor(ThreadMatrixC{}, "c_thread_mtx: ");

            printf("%u %u, %u %u %u, %u %u\n",
                   get_block_1d_id(),
                   get_thread_local_1d_id(),
                   c_thread_mtx_index.batch,
                   c_thread_mtx_index.row,
                   c_thread_mtx_index.col,
                   mMyThreadOffsetA,
                   mMyThreadOffsetB);
        }
#endif
    }

    __device__ MatrixIndex GetBeginOfThreadMatrixC(unsigned thread_id) const
    {
        constexpr unsigned BatchThreadWork = BatchSize / BatchPerThread;

        constexpr unsigned ThreadPerLevel1Cluster =
            MLevel0Cluster * NLevel0Cluster * MLevel1Cluster * NLevel1Cluster;

        constexpr unsigned ThreadPerLevel0Cluster = MLevel0Cluster * NLevel0Cluster;

        unsigned batch_work_id = thread_id / ThreadPerLevel1Cluster;
        unsigned cluster_id    = thread_id - batch_work_id * ThreadPerLevel1Cluster;

        unsigned level1_id   = cluster_id / ThreadPerLevel0Cluster;
        unsigned level1_m_id = level1_id / NLevel1Cluster;
        unsigned level1_n_id = level1_id % NLevel1Cluster;

        unsigned level0_id   = cluster_id % ThreadPerLevel0Cluster;
        unsigned level0_m_id = level0_id / NLevel0Cluster;
        unsigned level0_n_id = level0_id % NLevel0Cluster;

        constexpr unsigned MPerLevel0Cluster = MPerThreadSubC * MLevel0Cluster;
        constexpr unsigned NPerLevel0Cluster = NPerThreadSubC * NLevel0Cluster;

        return MatrixIndex{batch_work_id * BatchPerThread,
                           level1_m_id * MPerLevel0Cluster + level0_m_id * MPerThreadSubC,
                           level1_n_id * NPerLevel0Cluster + level0_n_id * NPerThreadSubC};
    }

    // this should be optimized away if input is known
    __device__ static MatrixIndex
    GetDistanceFromBeginOfThreadMatrixC(unsigned batch_in_c, unsigned m_in_c, unsigned n_in_c)
    {
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

        constexpr unsigned MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr unsigned NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        unsigned m_repeat = m_in_c / MPerThreadSubC;
        unsigned n_repeat = n_in_c / NPerThreadSubC;

        unsigned m_in_sub_c = m_in_c % MPerThreadSubC;
        unsigned n_in_sub_c = n_in_c % NPerThreadSubC;

        return MatrixIndex{batch_in_c,
                           m_repeat * MPerLevel1Cluster + m_in_sub_c,
                           n_repeat * NPerLevel1Cluster + n_in_sub_c};
    }

    template <class FloatA, class FloatB, class FloatC, class Accumulator>
    __device__ void Run(const FloatA* __restrict__ p_a_block,
                        const FloatB* __restrict__ p_b_block,
                        FloatC* __restrict__ p_c_thread,
                        Accumulator f_accum) const
    {
        constexpr auto True  = integral_constant<bool, true>{};
        constexpr auto False = integral_constant<bool, false>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr unsigned KPerBlock = a_block_mtx.NRow(); // A is transposed

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        // thread A, B for GEMM
        //   A is transposed, b is not
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        // thread A-sub, B-sub for copy
        constexpr auto a_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{}, Number<MPerThread>{});

        constexpr auto b_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

        constexpr unsigned MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr unsigned NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

// loop over k
#pragma unroll
        for(unsigned k_begin = 0; k_begin < KPerBlock; k_begin += KPerThreadLoop)
        {
// read first batch of A, B
//   copy A-sub to form A
#pragma unroll
            for(unsigned m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
            {
                threadwise_matrix_copy(
                    a_block_mtx,
                    p_a_block + a_block_mtx.Get1dIndex(k_begin, m_repeat * MPerLevel1Cluster) +
                        mMyThreadOffsetA,
                    a_thread_mtx,
                    p_a_thread + a_thread_mtx.Get1dIndex(0, m_repeat * MPerThreadSubC),
                    a_thread_sub_mtx.GetLengths());
            }

//   copy B-sub to form B
#pragma unroll
            for(unsigned n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
            {
                threadwise_matrix_copy(
                    b_block_mtx,
                    p_b_block + b_block_mtx.Get1dIndex(k_begin, n_repeat * NPerLevel1Cluster) +
                        mMyThreadOffsetB,
                    b_thread_mtx,
                    p_b_thread + b_thread_mtx.Get1dIndex(0, n_repeat * NPerThreadSubC),
                    b_thread_sub_mtx.GetLengths());
            }

// loop over batch
#pragma unroll
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
#pragma unroll
                    for(unsigned m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
                    {
                        threadwise_matrix_copy(
                            a_block_mtx,
                            p_a_block +
                                a_block_mtx.Get1dIndex(k_begin, m_repeat * MPerLevel1Cluster) +
                                (ib + 1) * BlockMatrixStrideA + mMyThreadOffsetA,
                            a_thread_mtx,
                            p_a_thread + a_thread_mtx.Get1dIndex(0, m_repeat * MPerThreadSubC),
                            a_thread_sub_mtx.GetLengths());
                    }
                }

                if(BlockMatrixStrideB != 0)
                {
#pragma unroll
                    for(unsigned n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
                    {
                        threadwise_matrix_copy(
                            b_block_mtx,
                            p_b_block +
                                b_block_mtx.Get1dIndex(k_begin, n_repeat * NPerLevel1Cluster) +
                                (ib + 1) * BlockMatrixStrideB + mMyThreadOffsetB,
                            b_thread_mtx,
                            p_b_thread + b_thread_mtx.Get1dIndex(0, n_repeat * NPerThreadSubC),
                            b_thread_sub_mtx.GetLengths());
                    }
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

    template <class BlockMatrixC, unsigned BlockMatrixStrideC, class FloatC>
    __device__ void CopyThreadMatrixCToBlockMatrixC(const FloatC* __restrict__ p_c_thread,
                                                    FloatC* __restrict__ p_c_block) const
    {
        constexpr auto c_block_mtx  = BlockMatrixC{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        constexpr auto c_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<MPerThreadSubC>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        constexpr unsigned MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr unsigned NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

        const auto c_thread_mtx_begin = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        const unsigned c_thread_offset =
            c_thread_mtx_begin.batch * BlockMatrixStrideC +
            c_block_mtx.Get1dIndex(c_thread_mtx_begin.row, c_thread_mtx_begin.col);

        for(unsigned m_repeat = 0; m_repeat, MRepeat; ++m_repeat)
        {
            for(unsigned n_repeat = 0; n_repeat, NRepeat; ++n_repeat)
            {
                threadwise_matrix_copy(
                    c_thread_sub_mtx,
                    p_c_thread +
                        c_thread_sub_mtx.Get1dIndex(m_repeat * MPerLevel1Cluster,
                                                    n_repeat * NPerLevel1Cluster),
                    c_block_mtx,
                    p_c_block +
                        c_block_mtx.Get1dIndex(m_repeat * MPerLevel1Cluster,
                                               n_repeat * NPerLevel1Cluster) +
                        c_thread_offset,
                    c_thread_sub_mtx.GetLengths());
            }
        }
    }
};

template <unsigned BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class ThreadMatrixC,
          bool TransA,
          bool TransB,
          bool TransC,
          unsigned KPerThreadLoop,
          unsigned MThreadPerCluster,
          unsigned NThreadPerCluster,
          bool DistributeThreadAlongColumnFirst>
struct BlockwiseGemmBlockABlockBThreadC
{
    unsigned mMyThreadOffsetA = 0;
    unsigned mMyThreadOffsetB = 0;

    struct MatrixIndex
    {
        unsigned row;
        unsigned col;
    };

    __device__ BlockwiseGemmBlockABlockBThreadC()
    {
        constexpr auto a_block_mtx = BlockMatrixA{};
        constexpr auto b_block_mtx = BlockMatrixB{};

        const auto c_thread_mtx_index = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        mMyThreadOffsetA = (!TransA) ? a_block_mtx.Get1dIndex(c_thread_mtx_index.row, 0)
                                     : a_block_mtx.Get1dIndex(0, c_thread_mtx_index.row);

        mMyThreadOffsetB = (!TransB) ? b_block_mtx.Get1dIndex(0, c_thread_mtx_index.col)
                                     : b_block_mtx.Get1dIndex(c_thread_mtx_index.col, 0);

#if 0
        if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
        {
            print_ConstantMatrixDescriptor(BlockMatrixA{}, "a_block_mtx: ");
            print_ConstantMatrixDescriptor(BlockMatrixB{}, "b_block_mtx: ");
            print_ConstantMatrixDescriptor(ThreadMatrixC{}, "c_thread_mtx: ");

            printf("%u %u, %u %u %u, %u %u\n",
                   get_block_1d_id(),
                   get_thread_local_1d_id(),
                   c_thread_mtx_index.batch,
                   c_thread_mtx_index.row,
                   c_thread_mtx_index.col,
                   mMyThreadOffsetA,
                   mMyThreadOffsetB);
        }
#endif
    }

    __device__ MatrixIndex GetBeginOfThreadMatrixC(unsigned thread_id) const
    {

        if(TransA && (!TransB) && (!TransC))
        {
            constexpr auto a_block_mtx = BlockMatrixA{};
            constexpr auto b_block_mtx = BlockMatrixB{};

            static_assert(a_block_mtx.NRow() == b_block_mtx.NRow(),
                          "wrong! k dimension not consistent!");

            constexpr unsigned MPerBlock = a_block_mtx.NCol();
            constexpr unsigned NPerBlock = b_block_mtx.NCol();

            constexpr auto c_thread_mtx = ThreadMatrixC{};

            // divide thread work
            constexpr unsigned MPerThread = c_thread_mtx.NRow();
            constexpr unsigned NPerThread = c_thread_mtx.NCol();

            static_assert(MPerBlock % (MPerThread * MThreadPerCluster) == 0,
                          "MPerBlock % (MPerThread * MThreadPerCluster) != 0");

            static_assert(NPerBlock % (NPerThread * NThreadPerCluster) == 0,
                          "NPerBlock % (NPerThread * NThreadPerCluster) != 0");

            constexpr unsigned MClusterWork =
                (MPerBlock + MPerThread * MThreadPerCluster - 1) / (MPerThread * MThreadPerCluster);

            constexpr unsigned NClusterWork =
                (NPerBlock + NPerThread * NThreadPerCluster - 1) / (NPerThread * NThreadPerCluster);

            static_assert(BlockSize ==
                              (MClusterWork * MThreadPerCluster) *
                                  (NClusterWork * NThreadPerCluster),
                          "wrong! wrong BlockSize");

            if(DistributeThreadAlongColumnFirst)
            {
                const unsigned cluster_work_block_id =
                    thread_id / (MThreadPerCluster * NThreadPerCluster);

                const unsigned thread_work_cluster_id =
                    thread_id - cluster_work_block_id * (MThreadPerCluster * NThreadPerCluster);

                const unsigned m_cluster_work_block_id = cluster_work_block_id / NClusterWork;
                const unsigned n_cluster_work_block_id =
                    cluster_work_block_id - m_cluster_work_block_id * NClusterWork;

                const unsigned m_thread_work_cluster_id =
                    thread_work_cluster_id / NThreadPerCluster;
                const unsigned n_thread_work_cluster_id =
                    thread_work_cluster_id - m_thread_work_cluster_id * NThreadPerCluster;

#if 0
                if(get_block_1d_id() == 0)
                {
                    printf("%u %u, \t"
                           "MClusterWork %u MThreadPerCluster %u NClusterWork %u NThreadPerCluster %u \t"
                           "m_cluster_work_block_id %u n_cluster_work_block_id %u \t"
                           "m_thread_work_cluster_id %u n_thread_work_cluster_id %u \t"
                            "\n",
                            get_block_1d_id(), get_thread_local_1d_id(),
                            MClusterWork, MThreadPerCluster, NClusterWork, NThreadPerCluster,
                            m_cluster_work_block_id, n_cluster_work_block_id,
                            m_thread_work_cluster_id, n_thread_work_cluster_id);
                }
#endif

                return MatrixIndex{m_cluster_work_block_id * (MThreadPerCluster * MPerThread) +
                                       m_thread_work_cluster_id * MPerThread,
                                   n_cluster_work_block_id * (NThreadPerCluster * NPerThread) +
                                       n_thread_work_cluster_id * NPerThread};
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

    // this should be optimized away if input is known
    __device__ static MatrixIndex GetDistanceFromBeginOfThreadMatrixC(unsigned m_in_c,
                                                                      unsigned n_in_c)
    {
        return MatrixIndex{m_in_c, n_in_c};
    }

    template <class FloatA, class FloatB, class FloatC, class Accumulator>
    __device__ void Run(const FloatA* __restrict__ p_a_block,
                        const FloatB* __restrict__ p_b_block,
                        FloatC* __restrict__ p_c_thread,
                        Accumulator f_accum) const
    {
        if(TransA && (!TransB) && (!TransC))
        {
            constexpr auto True  = integral_constant<bool, true>{};
            constexpr auto False = integral_constant<bool, false>{};

            constexpr auto a_block_mtx  = BlockMatrixA{};
            constexpr auto b_block_mtx  = BlockMatrixB{};
            constexpr auto c_thread_mtx = ThreadMatrixC{};

            constexpr unsigned KPerBlock = a_block_mtx.NRow(); // A is transposed

            constexpr unsigned MPerThread = c_thread_mtx.NRow();
            constexpr unsigned NPerThread = c_thread_mtx.NCol();

            // a is transposed, b is not
            constexpr auto a_thread_mtx =
                make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

            constexpr auto b_thread_mtx =
                make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

            FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
            FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

            // loop over k
            for(unsigned k_begin = 0; k_begin < KPerBlock; k_begin += KPerThreadLoop)
            {
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

                threadwise_gemm(a_thread_mtx,
                                True,
                                p_a_thread,
                                b_thread_mtx,
                                False,
                                p_b_thread,
                                c_thread_mtx,
                                False,
                                p_c_thread,
                                f_accum);
            }
        }
    }
};

// if following number are power of 2, index calculation shall be greatly reduced:
//    MPerThreadSubC, NPerThreadSubC, MLevel0Cluster, NLevel0Cluster, MLevel1Cluster, NLevel1Cluster
template <unsigned BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class ThreadMatrixC,
          unsigned MPerThreadSubC,
          unsigned NPerThreadSubC,
          unsigned MLevel0Cluster,
          unsigned NLevel0Cluster,
          unsigned MLevel1Cluster,
          unsigned NLevel1Cluster,
          unsigned KPerThreadLoop>
struct BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2
{
    struct MatrixIndex
    {
        unsigned row;
        unsigned col;
    };

    unsigned mMyThreadOffsetA;
    unsigned mMyThreadOffsetB;

    __device__ BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2()
    {
        constexpr unsigned ThreadPerLevel1Cluster =
            MLevel0Cluster * NLevel0Cluster * MLevel1Cluster * NLevel1Cluster;

        static_assert(BlockSize == ThreadPerLevel1Cluster, "wrong! wrong blocksize\n");

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        static_assert(a_block_mtx.NRow() == b_block_mtx.NRow(),
                      "wrong! K dimension not consistent\n");

        constexpr unsigned M = a_block_mtx.NCol(); // A is transposed
        constexpr unsigned N = b_block_mtx.NCol();
        constexpr unsigned K = a_block_mtx.NRow();

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        static_assert((MPerThread % MPerThreadSubC == 0) && (NPerThread % NPerThreadSubC == 0),
                      "wrong! Cannot evenly divide thread work among repeat \n");

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

        static_assert((M % MRepeat == 0) && (N % NRepeat == 0),
                      "wrong! Cannot evenly divide work among repeat\n");

        constexpr unsigned MPerLevel1Cluster = M / MRepeat;
        constexpr unsigned NPerLevel1Cluster = N / NRepeat;

        static_assert((MPerLevel1Cluster % MLevel1Cluster == 0) &&
                          (NPerLevel1Cluster % NLevel1Cluster == 0),
                      "wrong! Cannot evenly divide work among Level1Cluster\n");

        constexpr unsigned MPerLevel0Cluster = MPerLevel1Cluster / MLevel1Cluster;
        constexpr unsigned NPerLevel0Cluster = NPerLevel1Cluster / NLevel1Cluster;

        static_assert((MPerLevel0Cluster % MLevel0Cluster == 0) &&
                          (NPerLevel0Cluster % NLevel0Cluster == 0),
                      "wrong! Cannot evenly divide work among Level0Cluster\n");

        static_assert((MPerThreadSubC == MPerLevel0Cluster / MLevel0Cluster) &&
                          (NPerThreadSubC == NPerLevel0Cluster / NLevel0Cluster),
                      "wrong! thread work size is wrong\n");

        auto c_thread_mtx_index = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        mMyThreadOffsetA = a_block_mtx.Get1dIndex(0, c_thread_mtx_index.row);
        mMyThreadOffsetB = b_block_mtx.Get1dIndex(0, c_thread_mtx_index.col);
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(unsigned thread_id)
    {
        constexpr unsigned ThreadPerLevel0Cluster = MLevel0Cluster * NLevel0Cluster;

        unsigned level1_id   = thread_id / ThreadPerLevel0Cluster;
        unsigned level1_m_id = level1_id / NLevel1Cluster;
        unsigned level1_n_id = level1_id % NLevel1Cluster;

        unsigned level0_id   = thread_id % ThreadPerLevel0Cluster;
        unsigned level0_m_id = level0_id / NLevel0Cluster;
        unsigned level0_n_id = level0_id % NLevel0Cluster;

        constexpr unsigned MPerLevel0Cluster = MPerThreadSubC * MLevel0Cluster;
        constexpr unsigned NPerLevel0Cluster = NPerThreadSubC * NLevel0Cluster;

        return MatrixIndex{level1_m_id * MPerLevel0Cluster + level0_m_id * MPerThreadSubC,
                           level1_n_id * NPerLevel0Cluster + level0_n_id * NPerThreadSubC};
    }

    // this should be optimized away if input is known
    __device__ static MatrixIndex GetDistanceFromBeginOfThreadMatrixC(unsigned m_in_c,
                                                                      unsigned n_in_c)
    {
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

        constexpr unsigned MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr unsigned NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        unsigned m_repeat = m_in_c / MPerThreadSubC;
        unsigned n_repeat = n_in_c / NPerThreadSubC;

        unsigned m_in_sub_c = m_in_c % MPerThreadSubC;
        unsigned n_in_sub_c = n_in_c % NPerThreadSubC;

        return MatrixIndex{m_repeat * MPerLevel1Cluster + m_in_sub_c,
                           n_repeat * NPerLevel1Cluster + n_in_sub_c};
    }

    template <class FloatA, class FloatB, class FloatC, class Accumulator>
    __device__ void Run(const FloatA* __restrict__ p_a_block,
                        const FloatB* __restrict__ p_b_block,
                        FloatC* __restrict__ p_c_thread,
                        Accumulator f_accum) const
    {
        constexpr auto True  = integral_constant<bool, true>{};
        constexpr auto False = integral_constant<bool, false>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr unsigned M = a_block_mtx.NCol();
        constexpr unsigned N = b_block_mtx.NCol();
        constexpr unsigned K = a_block_mtx.NRow();

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        // thread A, B for GEMM
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        // thread A-sub, B-sub for copy
        constexpr auto a_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{}, Number<MPerThread>{});

        constexpr auto b_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

        constexpr unsigned MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr unsigned NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

#pragma unroll
        // loop over k
        for(unsigned k_begin = 0; k_begin < K; k_begin += KPerThreadLoop)
        {
#pragma unroll
            // copy A-sub to form A
            for(unsigned m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
            {
                threadwise_matrix_copy(
                    a_block_mtx,
                    p_a_block + a_block_mtx.Get1dIndex(k_begin, m_repeat * MPerLevel1Cluster) +
                        mMyThreadOffsetA,
                    a_thread_mtx,
                    p_a_thread + a_thread_mtx.Get1dIndex(0, m_repeat * MPerThreadSubC),
                    a_thread_sub_mtx.GetLengths());
            }

#pragma unroll
            // copy B-sub to form B
            for(unsigned n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
            {
                threadwise_matrix_copy(
                    b_block_mtx,
                    p_b_block + b_block_mtx.Get1dIndex(k_begin, n_repeat * NPerLevel1Cluster) +
                        mMyThreadOffsetB,
                    b_thread_mtx,
                    p_b_thread + b_thread_mtx.Get1dIndex(0, n_repeat * NPerThreadSubC),
                    b_thread_sub_mtx.GetLengths());
            }

            // C = A * B
            threadwise_gemm(a_thread_mtx,
                            True,
                            p_a_thread,
                            b_thread_mtx,
                            False,
                            p_b_thread,
                            c_thread_mtx,
                            False,
                            p_c_thread,
                            f_accum);
        }
    }

    template <class FloatA, class FloatB, class FloatC, class Accumulator>
    __device__ void Run_RegisterDoubleBuffer(FloatA* const p_a_block,
                                             FloatB* const p_b_block,
                                             FloatC* p_c_thread,
                                             Accumulator f_accum) const
    {
        constexpr auto True  = integral_constant<bool, true>{};
        constexpr auto False = integral_constant<bool, false>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr unsigned M = a_block_mtx.NCol();
        constexpr unsigned N = b_block_mtx.NCol();
        constexpr unsigned K = a_block_mtx.NRow();

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        // thread A, B for GEMM
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        // thread A-sub, B-sub for copy
        constexpr auto a_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{}, Number<MPerThread>{});

        constexpr auto b_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        // register
        FloatA p_a_thread_0[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread_0[b_thread_mtx.GetElementSpace()];

        FloatA p_a_thread_1[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread_1[b_thread_mtx.GetElementSpace()];

        constexpr unsigned MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr unsigned NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

// preload A, B
#pragma unroll
        for(unsigned m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
        { // copy A-sub to form A
            threadwise_matrix_copy(a_block_mtx,
                                   p_a_block + mMyThreadOffsetA + m_repeat * MPerLevel1Cluster,
                                   a_thread_sub_mtx,
                                   p_a_thread_0 + m_repeat * MPerThreadSubC,
                                   a_thread_sub_mtx.GetLengths());
        }

#pragma unroll
        for(unsigned n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
        { // copy B-sub to form B
            threadwise_matrix_copy(b_block_mtx,
                                   p_b_block + mMyThreadOffsetB + n_repeat * NPerLevel1Cluster,
                                   b_thread_sub_mtx,
                                   p_b_thread_0 + n_repeat * NPerThreadSubC,
                                   b_thread_sub_mtx.GetLengths());
        }

        bool even_loop = true;

#pragma unroll
        for(unsigned k_begin = 0; k_begin + KPerThreadLoop < K;
            k_begin += KPerThreadLoop, even_loop = !even_loop)
        { // loop over k
            FloatA* p_a_thread_now = even_loop ? p_a_thread_0 : p_a_thread_1;
            FloatB* p_b_thread_now = even_loop ? p_b_thread_0 : p_b_thread_1;

            FloatA* p_a_thread_next = even_loop ? p_a_thread_1 : p_a_thread_0;
            FloatB* p_b_thread_next = even_loop ? p_b_thread_1 : p_b_thread_0;

// preload next A, B
#pragma unroll
            for(unsigned m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
            { // copy A-sub to form A
                threadwise_matrix_copy(a_block_mtx,
                                       p_a_block + mMyThreadOffsetA +
                                           (k_begin + 1) * a_block_mtx.RowStride() +
                                           m_repeat * MPerLevel1Cluster,
                                       a_thread_sub_mtx,
                                       p_a_thread_next + m_repeat * MPerThreadSubC,
                                       a_thread_sub_mtx.GetLengths());
            }

#pragma unroll
            for(unsigned n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
            { // copy B-sub to form B
                threadwise_matrix_copy(b_block_mtx,
                                       p_b_block + mMyThreadOffsetB +
                                           (k_begin + 1) * b_block_mtx.RowStride() +
                                           n_repeat * NPerLevel1Cluster,
                                       b_thread_sub_mtx,
                                       p_b_thread_next + n_repeat * NPerThreadSubC,
                                       b_thread_sub_mtx.GetLengths());
            }

            // C = A * B
            threadwise_gemm(a_thread_mtx,
                            True,
                            p_a_thread_now,
                            b_thread_mtx,
                            False,
                            p_b_thread_now,
                            c_thread_mtx,
                            False,
                            p_c_thread,
                            f_accum);
        }

        // last loop
        {
            FloatA* p_a_thread_now = even_loop ? p_a_thread_0 : p_a_thread_1;
            FloatB* p_b_thread_now = even_loop ? p_b_thread_0 : p_b_thread_1;

            // C = A * B
            threadwise_gemm(a_thread_mtx,
                            True,
                            p_a_thread_now,
                            b_thread_mtx,
                            False,
                            p_b_thread_now,
                            c_thread_mtx,
                            False,
                            p_c_thread,
                            f_accum);
        }
    }

    template <class FloatA, class FloatB, class FloatC, class Accumulator>
    __device__ void Run_v2(const FloatA* __restrict__ p_a_block,
                           const FloatB* __restrict__ p_b_block,
                           FloatC* __restrict__ p_c_thread,
                           Accumulator f_accum) const
    {
        constexpr auto True  = integral_constant<bool, true>{};
        constexpr auto False = integral_constant<bool, false>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr unsigned M = a_block_mtx.NCol();
        constexpr unsigned N = b_block_mtx.NCol();
        constexpr unsigned K = a_block_mtx.NRow();

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        // thread A-sub, B-sub, C-sub
        constexpr auto a_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{}, Number<MPerThread>{});

        constexpr auto b_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        constexpr auto c_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<MPerThreadSubC>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        // thread A, B
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

        constexpr unsigned MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr unsigned NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

#pragma unroll
        // loop over k
        for(unsigned k_begin = 0; k_begin < K; k_begin += KPerThreadLoop)
        {
            // C-sub(s) in first row-wise subblock of C
            {
                //   copy first A-sub
                threadwise_matrix_copy(a_block_mtx,
                                       p_a_block + a_block_mtx.Get1dIndex(k_begin, 0) +
                                           mMyThreadOffsetA,
                                       a_thread_mtx,
                                       p_a_thread,
                                       a_thread_sub_mtx.GetLengths());

                //   copy first B-sub
                threadwise_matrix_copy(b_block_mtx,
                                       p_b_block + b_block_mtx.Get1dIndex(k_begin, 0) +
                                           mMyThreadOffsetB,
                                       b_thread_mtx,
                                       p_b_thread,
                                       b_thread_sub_mtx.GetLengths());

                //   do first sub GEMM
                threadwise_gemm(a_thread_sub_mtx,
                                True,
                                p_a_thread,
                                b_thread_sub_mtx,
                                False,
                                p_b_thread,
                                c_thread_sub_mtx,
                                False,
                                p_c_thread,
                                f_accum);

#pragma unroll
                //   copy next B-sub, and do GEMM
                for(unsigned n_repeat = 1; n_repeat < NRepeat; ++n_repeat)
                {
                    threadwise_matrix_copy(
                        b_block_mtx,
                        p_b_block + b_block_mtx.Get1dIndex(k_begin, n_repeat * NPerLevel1Cluster) +
                            mMyThreadOffsetB,
                        b_thread_mtx,
                        p_b_thread + b_thread_mtx.Get1dIndex(0, n_repeat * NPerThreadSubC),
                        b_thread_sub_mtx.GetLengths());

                    threadwise_gemm(
                        a_thread_sub_mtx,
                        True,
                        p_a_thread,
                        b_thread_sub_mtx,
                        False,
                        p_b_thread + b_thread_mtx.Get1dIndex(0, n_repeat * NPerThreadSubC),
                        c_thread_sub_mtx,
                        False,
                        p_c_thread + c_thread_mtx.Get1dIndex(0, n_repeat * NPerThreadSubC),
                        f_accum);
                }

#pragma unroll
                // loop over rest of row-wise subblock
                //   all B-sub(s) has been copied, so only A-sub(s) need to be copied
                for(unsigned m_repeat = 1; m_repeat < MRepeat; ++m_repeat)
                {
                    // copy a A-sub
                    threadwise_matrix_copy(
                        a_block_mtx,
                        p_a_block + a_block_mtx.Get1dIndex(k_begin, m_repeat * MPerLevel1Cluster) +
                            mMyThreadOffsetA,
                        a_thread_mtx,
                        p_a_thread + a_thread_mtx.Get1dIndex(0, m_repeat * MPerThreadSubC),
                        a_thread_sub_mtx.GetLengths());

                    // do some GEMMs
                    for(unsigned n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
                    {
                        threadwise_gemm(
                            a_thread_sub_mtx,
                            True,
                            p_a_thread + a_thread_mtx.Get1dIndex(0, m_repeat * MPerThreadSubC),
                            b_thread_sub_mtx,
                            False,
                            p_b_thread + b_thread_mtx.Get1dIndex(0, n_repeat * NPerThreadSubC),
                            c_thread_sub_mtx,
                            False,
                            p_c_thread +
                                c_thread_mtx.Get1dIndex(m_repeat * MPerThreadSubC,
                                                        n_repeat * NPerThreadSubC),
                            f_accum);
                    }
                }
            }
        }
    }
};
