#ifndef CK_BLOCKWISE_BATCHED_GEMM_HPP
#define CK_BLOCKWISE_BATCHED_GEMM_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "threadwise_gemm.hpp"

#ifndef CK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM
#define CK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM 1
#endif

namespace ck {

template <index_t BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class ThreadMatrixC,
          index_t BlockMatrixStrideA,
          index_t BlockMatrixStrideB,
          index_t ThreadMatrixStrideC,
          index_t BatchSize,
          index_t MPerThreadSubC,
          index_t NPerThreadSubC,
          index_t MLevel0Cluster,
          index_t NLevel0Cluster,
          index_t MLevel1Cluster,
          index_t NLevel1Cluster,
          index_t KPerThreadLoop,
          index_t BatchPerThread,
          index_t DataPerReadA,
          index_t DataPerReadB>
struct BlockwiseBatchGemmBlockABlockBThreadCTransANormalBNormalC_V2
{
    index_t mMyThreadOffsetA = 0;
    index_t mMyThreadOffsetB = 0;

    struct MatrixIndex
    {
        index_t batch;
        index_t row;
        index_t col;
    };

    __device__ BlockwiseBatchGemmBlockABlockBThreadCTransANormalBNormalC_V2()
    {
        static_assert(BatchSize % BatchPerThread == 0,
                      "wrong! BatchSize is not dividable by BatchPerThread");

        constexpr index_t BatchThreadWork = BatchSize / BatchPerThread;

        constexpr index_t ThreadPerLevel1Cluster =
            MLevel0Cluster * NLevel0Cluster * MLevel1Cluster * NLevel1Cluster;

        static_assert(BlockSize == BatchThreadWork * ThreadPerLevel1Cluster,
                      "wrong! wrong blocksize\n");

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        static_assert(a_block_mtx.NRow() == b_block_mtx.NRow(),
                      "wrong! K dimension not consistent\n");

        constexpr index_t M = a_block_mtx.NCol(); // A is transposed
        constexpr index_t N = b_block_mtx.NCol();

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

        const auto c_thread_mtx_index = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        mMyThreadOffsetA = c_thread_mtx_index.batch * BlockMatrixStrideA +
                           a_block_mtx.GetOffsetFromMultiIndex(0, c_thread_mtx_index.row);

        mMyThreadOffsetB = c_thread_mtx_index.batch * BlockMatrixStrideB +
                           b_block_mtx.GetOffsetFromMultiIndex(0, c_thread_mtx_index.col);
    }

    __device__ MatrixIndex GetBeginOfThreadMatrixC(index_t thread_id) const
    {
        constexpr index_t ThreadPerLevel1Cluster =
            MLevel0Cluster * NLevel0Cluster * MLevel1Cluster * NLevel1Cluster;

        constexpr index_t ThreadPerLevel0Cluster = MLevel0Cluster * NLevel0Cluster;

        index_t batch_work_id = thread_id / ThreadPerLevel1Cluster;
        index_t cluster_id    = thread_id - batch_work_id * ThreadPerLevel1Cluster;

        index_t level1_id   = cluster_id / ThreadPerLevel0Cluster;
        index_t level1_m_id = level1_id / NLevel1Cluster;
        index_t level1_n_id = level1_id % NLevel1Cluster;

        index_t level0_id   = cluster_id % ThreadPerLevel0Cluster;
        index_t level0_m_id = level0_id / NLevel0Cluster;
        index_t level0_n_id = level0_id % NLevel0Cluster;

        constexpr index_t MPerLevel0Cluster = MPerThreadSubC * MLevel0Cluster;
        constexpr index_t NPerLevel0Cluster = NPerThreadSubC * NLevel0Cluster;

        return MatrixIndex{batch_work_id * BatchPerThread,
                           level1_m_id * MPerLevel0Cluster + level0_m_id * MPerThreadSubC,
                           level1_n_id * NPerLevel0Cluster + level0_n_id * NPerThreadSubC};
    }

    // this should be optimized away because input will be known at compile time
    __device__ static MatrixIndex
    GetDistanceFromBeginOfThreadMatrixC(index_t batch_in_c, index_t m_in_c, index_t n_in_c)
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

        return MatrixIndex{batch_in_c,
                           m_repeat * MPerLevel1Cluster + m_in_sub_c,
                           n_repeat * NPerLevel1Cluster + n_in_sub_c};
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run_source(const FloatA* __restrict__ p_a_block,
                               const FloatB* __restrict__ p_b_block,
                               FloatC* __restrict__ p_c_thread) const
    {
        constexpr auto True  = integral_constant<bool, true>{};
        constexpr auto False = integral_constant<bool, false>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t KPerBlock = a_block_mtx.NRow(); // A is transposed

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

        // thread A, B for GEMM
        //   A is transposed, b is not
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor_packed(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor_packed(Number<KPerThreadLoop>{}, Number<NPerThread>{});

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

// loop over k
#pragma unroll
        for(index_t k_begin = 0; k_begin < KPerBlock; k_begin += KPerThreadLoop)
        {
// loop over batch
#pragma unroll
            for(index_t ib = 0; ib < BatchPerThread; ++ib)
            {
                // read next batch of a, b
                if(BlockMatrixStrideA != 0 or ib == 0)
                {
#pragma unroll
                    for(index_t m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
                    {
                        threadwise_matrix_copy(a_block_mtx,
                                               p_a_block +
                                                   a_block_mtx.GetOffsetFromMultiIndex(
                                                       k_begin, m_repeat * MPerLevel1Cluster) +
                                                   ib * BlockMatrixStrideA + mMyThreadOffsetA,
                                               a_thread_mtx,
                                               p_a_thread + a_thread_mtx.GetOffsetFromMultiIndex(
                                                                0, m_repeat * MPerThreadSubC),
                                               a_thread_sub_mtx.GetLengths(),
                                               Number<DataPerReadA>{});
                    }
                }

                if(BlockMatrixStrideB != 0 or ib == 0)
                {
#pragma unroll
                    for(index_t n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
                    {
                        threadwise_matrix_copy(b_block_mtx,
                                               p_b_block +
                                                   b_block_mtx.GetOffsetFromMultiIndex(
                                                       k_begin, n_repeat * NPerLevel1Cluster) +
                                                   ib * BlockMatrixStrideB + mMyThreadOffsetB,
                                               b_thread_mtx,
                                               p_b_thread + b_thread_mtx.GetOffsetFromMultiIndex(
                                                                0, n_repeat * NPerThreadSubC),
                                               b_thread_sub_mtx.GetLengths(),
                                               Number<DataPerReadB>{});
                    }
                }

                threadwise_gemm(a_thread_mtx,
                                True,
                                p_a_thread,
                                b_thread_mtx,
                                False,
                                p_b_thread,
                                c_thread_mtx,
                                False,
                                p_c_thread + ib * ThreadMatrixStrideC);
            }
        }
    }

#if CK_USE_AMD_INLINE_ASM
    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run_amd_asm(const FloatA* __restrict__ p_a_block,
                                const FloatB* __restrict__ p_b_block,
                                FloatC* __restrict__ p_c_thread) const
    {
        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t K = a_block_mtx.NRow(); // A is transposed

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

        // thread A, B for GEMM
        //   A is transposed, b is not
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor_packed(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor_packed(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        // thread A-sub, B-sub for copy
        constexpr auto a_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{}, Number<MPerThread>{});

        constexpr auto b_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        // assertion for inline asm
        static_assert(is_same<FloatA, float>{} && is_same<FloatB, float>{} &&
                          is_same<FloatC, float>{},
                      "Run_amd_asm only deal with float\n");

        static_assert(MPerThreadSubC == 4 && NPerThreadSubC == 4 && KPerThreadLoop == 1 &&
                          MPerThread == 8 && NPerThread == 8,
                      "Run_amd_asm cannot deal with this GEMM shape yet\n");

        static_assert(DataPerReadA == 4 && DataPerReadB == 4, "Run_amd_asm only do float4 read\n");

        static_assert(BlockMatrixStrideA == 0 && BatchPerThread == 1,
                      "Run_amd_asm can only deal with BlockMatrixStrideA == 0 && BatchPerThread == "
                      "1 for now\n");

        using Float4 = vector_type<float, 4>::type;

        Float4* reg_a = (Float4*)(p_a_thread);
        Float4* reg_b = (Float4*)(p_b_thread);
        Float4* reg_c = (Float4*)(p_c_thread);

        reg_a[0] = *reinterpret_cast<const Float4*>(&p_a_block[mMyThreadOffsetA]);
        reg_b[0] = *reinterpret_cast<const Float4*>(&p_b_block[mMyThreadOffsetB]);
        reg_b[1] = *reinterpret_cast<const Float4*>(
            &p_b_block[b_block_mtx.GetOffsetFromMultiIndex(0, NPerLevel1Cluster) +
                       mMyThreadOffsetB]);
        reg_a[1] = *reinterpret_cast<const Float4*>(
            &p_a_block[a_block_mtx.GetOffsetFromMultiIndex(0, MPerLevel1Cluster) +
                       mMyThreadOffsetA]);
        outerProduct4x4(reg_a[0], reg_b[0], reg_c[0], reg_c[2], reg_c[4], reg_c[6]);
        outerProduct4x4(reg_a[0], reg_b[1], reg_c[1], reg_c[3], reg_c[5], reg_c[7]);

#pragma unroll
        for(index_t k = 1; k < K; ++k)
        {
            reg_a[0] = *reinterpret_cast<const Float4*>(
                &p_a_block[a_block_mtx.GetOffsetFromMultiIndex(k, 0) + mMyThreadOffsetA]);
            outerProduct4x4(reg_a[1], reg_b[0], reg_c[8], reg_c[10], reg_c[12], reg_c[14]);
            reg_b[0] = *reinterpret_cast<const Float4*>(
                &p_b_block[b_block_mtx.GetOffsetFromMultiIndex(k, 0) + mMyThreadOffsetB]);
            outerProduct4x4(reg_a[1], reg_b[1], reg_c[9], reg_c[11], reg_c[13], reg_c[15]);
            reg_b[1] = *reinterpret_cast<const Float4*>(
                &p_b_block[b_block_mtx.GetOffsetFromMultiIndex(k, NPerLevel1Cluster) +
                           mMyThreadOffsetB]);
            reg_a[1] = *reinterpret_cast<const Float4*>(
                &p_a_block[a_block_mtx.GetOffsetFromMultiIndex(k, MPerLevel1Cluster) +
                           mMyThreadOffsetA]);
            outerProduct4x4(reg_a[0], reg_b[0], reg_c[0], reg_c[2], reg_c[4], reg_c[6]);
            outerProduct4x4(reg_a[0], reg_b[1], reg_c[1], reg_c[3], reg_c[5], reg_c[7]);
        }
        outerProduct4x4(reg_a[1], reg_b[0], reg_c[8], reg_c[10], reg_c[12], reg_c[14]);
        outerProduct4x4(reg_a[1], reg_b[1], reg_c[9], reg_c[11], reg_c[13], reg_c[15]);
    }
#endif

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA* __restrict__ p_a_block,
                        const FloatB* __restrict__ p_b_block,
                        FloatC* __restrict__ p_c_thread) const

    {
#if CK_USE_AMD_INLINE_ASM && CK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM
        Run_amd_asm(p_a_block, p_b_block, p_c_thread);
#else
        Run_source(p_a_block, p_b_block, p_c_thread);
#endif
    }

    template <class BlockMatrixC, index_t BlockMatrixStrideC, class FloatC>
    __device__ void CopyThreadMatrixCToBlockMatrixC(const FloatC* __restrict__ p_c_thread,
                                                    FloatC* __restrict__ p_c_block) const
    {
        constexpr auto c_block_mtx  = BlockMatrixC{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr index_t MPerThread = c_thread_mtx.NRow();
        constexpr index_t NPerThread = c_thread_mtx.NCol();

        constexpr auto c_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<MPerThreadSubC>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        constexpr index_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr index_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        const auto c_thread_mtx_begin = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        const index_t c_thread_offset =
            c_thread_mtx_begin.batch * BlockMatrixStrideC +
            c_block_mtx.GetOffsetFromMultiIndex(c_thread_mtx_begin.row, c_thread_mtx_begin.col);

        for(index_t m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
        {
            for(index_t n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
            {
                threadwise_matrix_copy(
                    c_thread_sub_mtx,
                    p_c_thread + c_thread_sub_mtx.GetOffsetFromMultiIndex(
                                     m_repeat * MPerLevel1Cluster, n_repeat * NPerLevel1Cluster),
                    c_block_mtx,
                    p_c_block +
                        c_block_mtx.GetOffsetFromMultiIndex(m_repeat * MPerLevel1Cluster,
                                                            n_repeat * NPerLevel1Cluster) +
                        c_thread_offset,
                    c_thread_sub_mtx.GetLengths());
            }
        }
    }
};

} // namespace ck
#endif
