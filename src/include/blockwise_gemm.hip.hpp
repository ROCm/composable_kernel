#pragma once
#include "threadwise_gemm.hip.hpp"

template <index_t BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class ThreadMatrixC,
          bool TransA,
          bool TransB,
          bool TransC,
          index_t KPerThreadLoop,
          index_t MThreadPerCluster,
          index_t NThreadPerCluster,
          bool DistributeThreadAlongColumnFirst>
struct BlockwiseGemmBlockABlockBThreadC
{
    index_t mMyThreadOffsetA = 0;
    index_t mMyThreadOffsetB = 0;

    struct MatrixIndex
    {
        index_t row;
        index_t col;
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

    __device__ MatrixIndex GetBeginOfThreadMatrixC(index_t thread_id) const
    {

        if(TransA && (!TransB) && (!TransC))
        {
            constexpr auto a_block_mtx = BlockMatrixA{};
            constexpr auto b_block_mtx = BlockMatrixB{};

            static_assert(a_block_mtx.NRow() == b_block_mtx.NRow(),
                          "wrong! k dimension not consistent!");

            constexpr index_t MPerBlock = a_block_mtx.NCol();
            constexpr index_t NPerBlock = b_block_mtx.NCol();

            constexpr auto c_thread_mtx = ThreadMatrixC{};

            // divide thread work
            constexpr index_t MPerThread = c_thread_mtx.NRow();
            constexpr index_t NPerThread = c_thread_mtx.NCol();

            static_assert(MPerBlock % (MPerThread * MThreadPerCluster) == 0,
                          "MPerBlock % (MPerThread * MThreadPerCluster) != 0");

            static_assert(NPerBlock % (NPerThread * NThreadPerCluster) == 0,
                          "NPerBlock % (NPerThread * NThreadPerCluster) != 0");

            constexpr index_t MClusterWork =
                (MPerBlock + MPerThread * MThreadPerCluster - 1) / (MPerThread * MThreadPerCluster);

            constexpr index_t NClusterWork =
                (NPerBlock + NPerThread * NThreadPerCluster - 1) / (NPerThread * NThreadPerCluster);

            static_assert(BlockSize ==
                              (MClusterWork * MThreadPerCluster) *
                                  (NClusterWork * NThreadPerCluster),
                          "wrong! wrong BlockSize");

            if(DistributeThreadAlongColumnFirst)
            {
                const index_t cluster_work_block_id =
                    thread_id / (MThreadPerCluster * NThreadPerCluster);

                const index_t thread_work_cluster_id =
                    thread_id - cluster_work_block_id * (MThreadPerCluster * NThreadPerCluster);

                const index_t m_cluster_work_block_id = cluster_work_block_id / NClusterWork;
                const index_t n_cluster_work_block_id =
                    cluster_work_block_id - m_cluster_work_block_id * NClusterWork;

                const index_t m_thread_work_cluster_id = thread_work_cluster_id / NThreadPerCluster;
                const index_t n_thread_work_cluster_id =
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
    __device__ static MatrixIndex GetDistanceFromBeginOfThreadMatrixC(index_t m_in_c,
                                                                      index_t n_in_c)
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

            constexpr index_t KPerBlock = a_block_mtx.NRow(); // A is transposed

            constexpr index_t MPerThread = c_thread_mtx.NRow();
            constexpr index_t NPerThread = c_thread_mtx.NCol();

            // a is transposed, b is not
            constexpr auto a_thread_mtx =
                make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

            constexpr auto b_thread_mtx =
                make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

            FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
            FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

            // loop over k
            for(index_t k_begin = 0; k_begin < KPerBlock; k_begin += KPerThreadLoop)
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
template <index_t BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class ThreadMatrixC,
          index_t MPerThreadSubC,
          index_t NPerThreadSubC,
          index_t MLevel0Cluster,
          index_t NLevel0Cluster,
          index_t MLevel1Cluster,
          index_t NLevel1Cluster,
          index_t KPerThreadLoop>
struct BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2
{
    struct MatrixIndex
    {
        index_t row;
        index_t col;
    };

    index_t mMyThreadOffsetA;
    index_t mMyThreadOffsetB;

    __device__ BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2()
    {
        constexpr index_t ThreadPerLevel1Cluster =
            MLevel0Cluster * NLevel0Cluster * MLevel1Cluster * NLevel1Cluster;

        static_assert(BlockSize == ThreadPerLevel1Cluster, "wrong! wrong blocksize\n");

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        static_assert(a_block_mtx.NRow() == b_block_mtx.NRow(),
                      "wrong! K dimension not consistent\n");

        constexpr index_t M = a_block_mtx.NCol(); // A is transposed
        constexpr index_t N = b_block_mtx.NCol();
        constexpr index_t K = a_block_mtx.NRow();

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

        static_assert((MPerThread % MPerThreadSubC == 0) && (NPerThread % NPerThreadSubC == 0),
                      "wrong! Cannot evenly divide thread work among repeat \n");

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        static_assert((M % MRepeat == 0) && (N % NRepeat == 0),
                      "wrong! Cannot evenly divide work among repeat\n");

        constexpr index_t MPerLevel1Cluster = M / MRepeat;
        constexpr index_t NPerLevel1Cluster = N / NRepeat;

        static_assert((MPerLevel1Cluster % MLevel1Cluster == 0) &&
                          (NPerLevel1Cluster % NLevel1Cluster == 0),
                      "wrong! Cannot evenly divide work among Level1Cluster\n");

        constexpr index_t MPerLevel0Cluster = MPerLevel1Cluster / MLevel1Cluster;
        constexpr index_t NPerLevel0Cluster = NPerLevel1Cluster / NLevel1Cluster;

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

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t thread_id)
    {
        constexpr index_t ThreadPerLevel0Cluster = MLevel0Cluster * NLevel0Cluster;

        index_t level1_id   = thread_id / ThreadPerLevel0Cluster;
        index_t level1_m_id = level1_id / NLevel1Cluster;
        index_t level1_n_id = level1_id % NLevel1Cluster;

        index_t level0_id   = thread_id % ThreadPerLevel0Cluster;
        index_t level0_m_id = level0_id / NLevel0Cluster;
        index_t level0_n_id = level0_id % NLevel0Cluster;

        constexpr index_t MPerLevel0Cluster = MPerThreadSubC * MLevel0Cluster;
        constexpr index_t NPerLevel0Cluster = NPerThreadSubC * NLevel0Cluster;

        return MatrixIndex{level1_m_id * MPerLevel0Cluster + level0_m_id * MPerThreadSubC,
                           level1_n_id * NPerLevel0Cluster + level0_n_id * NPerThreadSubC};
    }

    // this should be optimized away if input is known
    __device__ static MatrixIndex GetDistanceFromBeginOfThreadMatrixC(index_t m_in_c,
                                                                      index_t n_in_c)
    {
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        index_t m_repeat = m_in_c / MPerThreadSubC;
        index_t n_repeat = n_in_c / NPerThreadSubC;

        index_t m_in_sub_c = m_in_c % MPerThreadSubC;
        index_t n_in_sub_c = n_in_c % NPerThreadSubC;

        return MatrixIndex{m_repeat * MPerLevel1Cluster + m_in_sub_c,
                           n_repeat * NPerLevel1Cluster + n_in_sub_c};
    }

    template <class FloatA, class FloatB, class FloatC, class Accumulator>
    __device__ void Run_asm(const FloatA* __restrict__ p_a_block,
                            const FloatB* __restrict__ p_b_block,
                            FloatC* __restrict__ p_c_thread,
                            Accumulator f_accum) const
    {
        constexpr auto True  = integral_constant<bool, true>{};
        constexpr auto False = integral_constant<bool, false>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t M = a_block_mtx.NCol();
        constexpr index_t N = b_block_mtx.NCol();
        constexpr index_t K = a_block_mtx.NRow();

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

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

        float p_thread[a_thread_mtx.GetElementSpace() + b_thread_mtx.GetElementSpace()];

        FloatA* p_a_thread = p_thread;
        FloatB* p_b_thread = p_thread + a_thread_mtx.GetElementSpace();

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        //auto a_src_index = a_block_mtx.Get1dIndex(k_begin, 0) + mMyThreadOffsetA;
        //auto b_src_index = b_block_mtx.Get1dIndex(k_begin, 0) + mMyThreadOffsetB;
        Float4* reg_a = (Float4*)(p_a_thread);
        Float4* reg_b = (Float4*)(p_b_thread);
        Float4* reg_c = (Float4*)(p_c_thread);
        void* a_loc = (void *)(p_a_block + mMyThreadOffsetA); 
        void* b_loc = (void *)(p_b_block + mMyThreadOffsetB); 
        // loop over k
        int k_chunk = 2;
#pragma unroll
        for(index_t k_begin = 0; k_begin < K; k_begin += KPerThreadLoop * k_chunk)
        {

#if 0
            ds_read_b128(reg_a[0], a_loc, 0);
            ds_read_b128(reg_a[1], a_loc, 256);
            ds_read_b128(reg_b[0], b_loc, 0);
            ds_read_b128(reg_b[1], b_loc, 128);

            lgkmcnt(0);

            outerProduct4x4(reg_a[0], reg_b[0], reg_c[0], reg_c[2], reg_c[4], reg_c[6]);
            outerProduct4x4(reg_a[0], reg_b[1], reg_c[1], reg_c[3], reg_c[5], reg_c[7]);
            outerProduct4x4(reg_a[1], reg_b[0], reg_c[8], reg_c[10], reg_c[12], reg_c[14]);
            outerProduct4x4(reg_a[1], reg_b[1], reg_c[9], reg_c[11], reg_c[13], reg_c[15]);
#else
            int k = k_begin;
            int lds_a_block_off = sizeof(Float) * M;
            int lds_b_block_off = sizeof(Float) * N;
            int lds_a_block_off_1 = MPerLevel1Cluster * sizeof(Float);
            int lds_b_block_off_1 = NPerLevel1Cluster * sizeof(Float);
            ds_read_b128(reg_a[0], a_loc, k * lds_a_block_off);
            ds_read_b128(reg_b[0], b_loc, k * lds_b_block_off);
            ds_read_b128(reg_b[1], b_loc, lds_b_block_off_1 + k * lds_b_block_off);
            ds_read_b128(reg_a[1], a_loc, lds_a_block_off_1 + k * lds_a_block_off);
            lgkmcnt(2);
            outerProduct4x4(reg_a[0], reg_b[0], reg_c[0], reg_c[2], reg_c[4], reg_c[6]);
            lgkmcnt(1);
            outerProduct4x4(reg_a[0], reg_b[1], reg_c[1], reg_c[3], reg_c[5], reg_c[7]);
            lgkmcnt(0);
            for(int i = 0; i < k_chunk - 1; i++)
            {
                k = k + 1;
                ds_read_b128(reg_a[0], a_loc, k * lds_a_block_off);
                outerProduct4x4(reg_a[1], reg_b[0], reg_c[8], reg_c[10], reg_c[12], reg_c[14]);
                ds_read_b128(reg_b[0], b_loc, k * lds_b_block_off);
                outerProduct4x4(reg_a[1], reg_b[1], reg_c[9], reg_c[11], reg_c[13], reg_c[15]);
                ds_read_b128(reg_b[1], b_loc, lds_b_block_off_1 + k * lds_b_block_off);
                ds_read_b128(reg_a[1], a_loc, lds_a_block_off_1 + k * lds_a_block_off);
                lgkmcnt(2);
                outerProduct4x4(reg_a[0], reg_b[0], reg_c[0], reg_c[2], reg_c[4], reg_c[6]);
                lgkmcnt(1);
                outerProduct4x4(reg_a[0], reg_b[1], reg_c[1], reg_c[3], reg_c[5], reg_c[7]);
                lgkmcnt(0);
            }
            outerProduct4x4(reg_a[1], reg_b[0], reg_c[8], reg_c[10], reg_c[12], reg_c[14]);
            outerProduct4x4(reg_a[1], reg_b[1], reg_c[9], reg_c[11], reg_c[13], reg_c[15]);
#endif
        }
    }

    template <class FloatA, class FloatB, class FloatC, class Accumulator>
    __device__ void Run(const FloatA* const __restrict__ p_a_block,
                        const FloatB* const __restrict__ p_b_block,
                        FloatC* const __restrict__ p_c_thread,
                        Accumulator f_accum) const
    {
        constexpr auto True  = integral_constant<bool, true>{};
        constexpr auto False = integral_constant<bool, false>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t M = a_block_mtx.NCol();
        constexpr index_t N = b_block_mtx.NCol();
        constexpr index_t K = a_block_mtx.NRow();

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

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

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        const FloatA* const p_a_block_thread_offset = p_a_block + mMyThreadOffsetA;

#pragma unroll
        // loop over k
        for(index_t k_begin = 0; k_begin < K; k_begin += KPerThreadLoop)
        {
#pragma unroll
            // copy A-sub to form A
            for(index_t m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
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
            for(index_t n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
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

        constexpr index_t M = a_block_mtx.NCol();
        constexpr index_t N = b_block_mtx.NCol();
        constexpr index_t K = a_block_mtx.NRow();

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

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

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

// preload A, B
#pragma unroll
        for(index_t m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
        { // copy A-sub to form A
            threadwise_matrix_copy(a_block_mtx,
                                   p_a_block + mMyThreadOffsetA + m_repeat * MPerLevel1Cluster,
                                   a_thread_sub_mtx,
                                   p_a_thread_0 + m_repeat * MPerThreadSubC,
                                   a_thread_sub_mtx.GetLengths());
        }

#pragma unroll
        for(index_t n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
        { // copy B-sub to form B
            threadwise_matrix_copy(b_block_mtx,
                                   p_b_block + mMyThreadOffsetB + n_repeat * NPerLevel1Cluster,
                                   b_thread_sub_mtx,
                                   p_b_thread_0 + n_repeat * NPerThreadSubC,
                                   b_thread_sub_mtx.GetLengths());
        }

        bool even_loop = true;

#pragma unroll
        for(index_t k_begin = 0; k_begin + KPerThreadLoop < K;
            k_begin += KPerThreadLoop, even_loop = !even_loop)
        { // loop over k
            FloatA* p_a_thread_now = even_loop ? p_a_thread_0 : p_a_thread_1;
            FloatB* p_b_thread_now = even_loop ? p_b_thread_0 : p_b_thread_1;

            FloatA* p_a_thread_next = even_loop ? p_a_thread_1 : p_a_thread_0;
            FloatB* p_b_thread_next = even_loop ? p_b_thread_1 : p_b_thread_0;

// preload next A, B
#pragma unroll
            for(index_t m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
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
            for(index_t n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
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

        constexpr index_t M = a_block_mtx.NCol();
        constexpr index_t N = b_block_mtx.NCol();
        constexpr index_t K = a_block_mtx.NRow();

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

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

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

#pragma unroll
        // loop over k
        for(index_t k_begin = 0; k_begin < K; k_begin += KPerThreadLoop)
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
                for(index_t n_repeat = 1; n_repeat < NRepeat; ++n_repeat)
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
                for(index_t m_repeat = 1; m_repeat < MRepeat; ++m_repeat)
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
                    for(index_t n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
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
