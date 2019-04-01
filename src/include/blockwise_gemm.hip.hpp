#pragma once
#include "threadwise_gemm.hip.hpp"

extern "C" __attribute__((address_space(3))) void* __to_local(void* p)[[hc]];

inline __device__ void outerProduct4x4(float4 &a, float4 &b, float4 &c0, float4 &c1, float4 &c2, float4 &c3) {
	asm volatile(
			"\n \
			v_mac_f32 %0, %4, %5 \n \
			v_mac_f32 %1, %4, %6 \n \
			v_mac_f32 %2, %4, %7 \n \
			v_mac_f32 %3, %4, %8 \n \
			"
			:
			:"v"(c0.x),"v"(c0.y),"v"(c0.z),"v"(c0.w), \
			"v"(a.x),"v"(b.x),"v"(b.y),"v"(b.z),"v"(b.w)
			);
	asm volatile(
			"\n \
			v_mac_f32 %0, %4, %5 \n \
			v_mac_f32 %1, %4, %6 \n \
			v_mac_f32 %2, %4, %7 \n \
			v_mac_f32 %3, %4, %8 \n \
			"
			:
			:"v"(c1.x),"v"(c1.y),"v"(c1.z),"v"(c1.w), \
			"v"(a.y),"v"(b.x),"v"(b.y),"v"(b.z),"v"(b.w)
			);
	asm volatile(
			"\n \
			v_mac_f32 %0, %4, %5 \n \
			v_mac_f32 %1, %4, %6 \n \
			v_mac_f32 %2, %4, %7 \n \
			v_mac_f32 %3, %4, %8 \n \
			"
			:
			:"v"(c2.x),"v"(c2.y),"v"(c2.z),"v"(c2.w), \
			"v"(a.z),"v"(b.x),"v"(b.y),"v"(b.z),"v"(b.w)
			);
	asm volatile(
			"\n \
			v_mac_f32 %0, %4, %5 \n \
			v_mac_f32 %1, %4, %6 \n \
			v_mac_f32 %2, %4, %7 \n \
			v_mac_f32 %3, %4, %8 \n \
			"
			:
			:"v"(c3.x),"v"(c3.y),"v"(c3.z),"v"(c3.w), \
			"v"(a.w),"v"(b.x),"v"(b.y),"v"(b.z),"v"(b.w)
			);
}


template<uint32_t cnt>
inline __device__ void lgkmcnt(){
	if(cnt == 0) {
		asm volatile("\n \
				s_waitcnt lgkmcnt(0) \n \
				"::);
	}
	if(cnt == 1) {
		asm volatile("\n \
				s_waitcnt lgkmcnt(1) \n \
				"::);
	}
	if(cnt == 2) {
		asm volatile("\n \
				s_waitcnt lgkmcnt(2) \n \
				"::);
	}
	if(cnt == 3) {
		asm volatile("\n \
				s_waitcnt lgkmcnt(3) \n \
				"::);
	}
	if(cnt == 4) {
		asm volatile("\n \
				s_waitcnt lgkmcnt(4) \n \
				"::);
	}
	if(cnt == 5) {
		asm volatile("\n \
				s_waitcnt lgkmcnt(5) \n \
				"::);
	}
	if(cnt == 6) {
		asm volatile("\n \
				s_waitcnt lgkmcnt(6) \n \
				"::);
	}
}


template<uint32_t off>
inline __device__ void shared_read_b128(float4 &a0, float4 &a1, float4 &b0, float4 &b1, uint32_t &ldsA, uint32_t &ldsB) {
	if(off == 0) {
		asm volatile("\n \
				ds_read_b128 %0, %4 offset:0 \n \
				ds_read_b128 %1, %4 offset:256 \n \
				ds_read_b128 %2, %5 offset:0 \n \
				ds_read_b128 %3, %5 offset:256 \n \
				"
				:"=v"(a0),"=v"(a1),"=v"(b0),"=v"(b1)
				:"v"(ldsA),"v"(ldsB));
	}
	if(off == 1*512) {
		asm volatile("\n \
				ds_read_b128 %0, %4 offset:1*512 \n \
				ds_read_b128 %1, %4 offset:1*512+256 \n \
				ds_read_b128 %2, %5 offset:1*512 \n \
				ds_read_b128 %3, %5 offset:1*512+256 \n \
				"
				:"=v"(a0),"=v"(a1),"=v"(b0),"=v"(b1)
				:"v"(ldsA),"v"(ldsB));
	}
	if(off == 2*512) {
		asm volatile("\n \
				ds_read_b128 %0, %4 offset:2*512 \n \
				ds_read_b128 %1, %4 offset:2*512+256 \n \
				ds_read_b128 %2, %5 offset:2*512 \n \
				ds_read_b128 %3, %5 offset:2*512+256 \n \
				"
				:"=v"(a0),"=v"(a1),"=v"(b0),"=v"(b1)
				:"v"(ldsA),"v"(ldsB));
	}
	if(off == 3*512) {
		asm volatile("\n \
				ds_read_b128 %0, %4 offset:3*512 \n \
				ds_read_b128 %1, %4 offset:3*512+256 \n \
				ds_read_b128 %2, %5 offset:3*512 \n \
				ds_read_b128 %3, %5 offset:3*512+256 \n \
				"
				:"=v"(a0),"=v"(a1),"=v"(b0),"=v"(b1)
				:"v"(ldsA),"v"(ldsB));
	}

	if(off == 4*512) {
		asm volatile("\n \
				ds_read_b128 %0, %4 offset:4*512 \n \
				ds_read_b128 %1, %4 offset:4*512+256 \n \
				ds_read_b128 %2, %5 offset:4*512 \n \
				ds_read_b128 %3, %5 offset:4*512+256 \n \
				"
				:"=v"(a0),"=v"(a1),"=v"(b0),"=v"(b1)
				:"v"(ldsA),"v"(ldsB));
	}
	if(off == 5*512) {
		asm volatile("\n \
				ds_read_b128 %0, %4 offset:5*512 \n \
				ds_read_b128 %1, %4 offset:5*512+256 \n \
				ds_read_b128 %2, %5 offset:5*512 \n \
				ds_read_b128 %3, %5 offset:5*512+256 \n \
				"
				:"=v"(a0),"=v"(a1),"=v"(b0),"=v"(b1)
				:"v"(ldsA),"v"(ldsB));
	}
	if(off == 6*512) {
		asm volatile("\n \
				ds_read_b128 %0, %4 offset:6*512 \n \
				ds_read_b128 %1, %4 offset:6*512+256 \n \
				ds_read_b128 %2, %5 offset:6*512 \n \
				ds_read_b128 %3, %5 offset:6*512+256 \n \
				"
				:"=v"(a0),"=v"(a1),"=v"(b0),"=v"(b1)
				:"v"(ldsA),"v"(ldsB));
	}
	if(off == 7*512) {
		asm volatile("\n \
				ds_read_b128 %0, %4 offset:7*512 \n \
				ds_read_b128 %1, %4 offset:7*512+256 \n \
				ds_read_b128 %2, %5 offset:7*512 \n \
				ds_read_b128 %3, %5 offset:7*512+256 \n \
				"
				:"=v"(a0),"=v"(a1),"=v"(b0),"=v"(b1)
				:"v"(ldsA),"v"(ldsB));
	}
}

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

    template <class FloatA, class FloatB, class FloatC, class Accumulator, index_t block_off>
    __device__ void Run_asm(const FloatA* __restrict__ p_a_block,
                            const FloatB* __restrict__ p_b_block,
                            FloatC* __restrict__ p_c_thread,
                            Accumulator f_accum,
                            Number<block_off>) const
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

#pragma unroll
        // loop over k
        for(index_t k_begin = 0; k_begin < K; k_begin += KPerThreadLoop)
        {
#if 1
            auto a_src_index = a_block_mtx.Get1dIndex(k_begin, 0) + mMyThreadOffsetA;
            auto b_src_index = b_block_mtx.Get1dIndex(k_begin, 0) + mMyThreadOffsetB;

            uint32_t a_loc = block_off + a_src_index;
            uint32_t b_loc = b_src_index;
            //const float4* a_loc = (const float4*)(p_b_block + block_off + a_src_index);
            //const float4* b_loc = (const float4*)(p_b_block + b_src_index);
            float4* reg         = (float4*)(p_thread);
			float4* c_v         = (float4*)(p_c_thread);

            //shared_read_b128<0>(reg[0], reg[1], reg[2], reg[3], a_loc, b_loc);
            //reg[0] = a_loc[0];
            //reg[1] = a_loc[16];
            //reg[2] = b_loc[0];
            //reg[3] = b_loc[8];

            //asm volatile("\n \
                    //ds_read_b128 %0, %1 \n \
                    //"
                    //: "=v"(reg[0])
                    //: "v"(a_loc)
                    //);

            //asm volatile("\n \
                    //ds_read_b128 %0, %1 \n \
                    //"
                    //: "=v"(reg[1])
                    //: "v"(a_loc + 256)
                    //);

            //asm volatile("\n \
                    //ds_read_b128 %0, %1 \n \
                    //"
                    //: "=v"(reg[2])
                    //: "v"(b_loc)
                    //);

            //asm volatile("\n \
                    //ds_read_b128 %0, %1\n \
                    //"
                    //: "=v"(reg[3])
                    //: "v"(b_loc + 128)
                    //);

            asm volatile("\n \
                    ds_read_b128 %0, %4 \n \
                    ds_read_b128 %1, %4 offset:16 \n \
                    ds_read_b128 %2, %5 \n \
                    ds_read_b128 %3, %5 offset:8 \n \
                    "
                    : "=v"(reg[0]), "=v"(reg[1]), "=v"(reg[2]), "=v"(reg[3])
                    : "v"(a_loc), "v"(b_loc)
                    );
            lgkmcnt<0>();

			outerProduct4x4(reg[0], reg[2], c_v[0], c_v[1], c_v[2], c_v[3]);
			outerProduct4x4(reg[0], reg[3], c_v[4], c_v[5], c_v[6], c_v[7]);
			outerProduct4x4(reg[1], reg[2], c_v[8], c_v[9], c_v[10], c_v[11]);
			outerProduct4x4(reg[1], reg[3], c_v[12], c_v[13], c_v[14], c_v[15]);

            //asm volatile("\n \
                    //ds_read_b32  %0, %16 \n \
                    //ds_read_b32  %1, %16 offset:1\n \
                    //ds_read_b32  %2, %16 offset:2\n \
                    //ds_read_b32  %3, %16 offset:3\n \
                    //ds_read_b32  %4, %17 \n \
                    //ds_read_b32  %5, %17 offset:1\n \
                    //ds_read_b32  %6, %17 offset:2\n \
                    //ds_read_b32  %7, %17 offset:3\n \
                    //ds_read_b32  %8, %18 \n \
                    //ds_read_b32  %9, %18 offset:1\n \
                    //ds_read_b32 %10, %18 offset:2\n \
                    //ds_read_b32 %11, %18 offset:3\n \
                    //ds_read_b32 %12, %19 \n \
                    //ds_read_b32 %13, %19 offset:1\n \
                    //ds_read_b32 %14, %19 offset:2\n \
                    //ds_read_b32 %15, %19 offset:3\n \
                    //s_waitcnt lgkmcnt(0)"
                    //:
                    //"=v"(p_a_thread[0]),
                    //"=v"(p_a_thread[1]),
                    //"=v"(p_a_thread[2]),
                    //"=v"(p_a_thread[3]),
                    //"=v"(p_a_thread[4]),
                    //"=v"(p_a_thread[5]),
                    //"=v"(p_a_thread[6]),
                    //"=v"(p_a_thread[7]),
                    //"=v"(p_b_thread[0]),
                    //"=v"(p_b_thread[1]),
                    //"=v"(p_b_thread[2]),
                    //"=v"(p_b_thread[3]),
                    //"=v"(p_b_thread[4]),
                    //"=v"(p_b_thread[5]),
                    //"=v"(p_b_thread[6]),
                    //"=v"(p_b_thread[7])
                         //:
                             //"v"(__to_local((void *)(&p_a_block[0]))),
                             //"v"(__to_local((void *)(&p_a_block[64]))),
                             //"v"(__to_local((void *)(&p_b_block[0]))),
                             //"v"(__to_local((void *)(&p_b_block[32])))
                                 //);

            //C = A * B
#else
            auto a_src_index = a_block_mtx.Get1dIndex(k_begin, 0) + mMyThreadOffsetA;
            auto b_src_index = b_block_mtx.Get1dIndex(k_begin, 0) + mMyThreadOffsetB;
            auto dst_index   = a_thread_sub_mtx.Get1dIndex(0, 0);

            const float4* a_loc = (const float4*)(p_a_block + a_src_index);
            const float4* b_loc = (const float4*)(p_b_block + b_src_index);
            float4* reg         = (float4*)(p_a_thread + dst_index);

            asm volatile("\n \
                                ds_read2_b64 %0, %84 offset1:1 \n \
                                ds_read2_b64 %1, %84 offset0:32 offset1:33 \n \
                                ds_read2_b64 %2, %85 offset1:1 \n \
                                ds_read2_b64 %3, %85 offset0:16 offset1:17 \n \
                                s_waitcnt lgkmcnt(0) \n \
                                v_mac_f32 %4, %68, %76 \n \
                                v_mac_f32 %5, %68, %77 \n \
                                v_mac_f32 %6, %68, %78 \n \
                                v_mac_f32 %7, %68, %79 \n \
                                v_mac_f32 %8, %68, %80 \n \
                                v_mac_f32 %9, %68, %81 \n \
                                v_mac_f32 %10, %68, %82 \n \
                                v_mac_f32 %11, %68, %83 \n \
                                v_mac_f32 %12, %69, %76 \n \
                                v_mac_f32 %13, %69, %77 \n \
                                v_mac_f32 %14, %69, %78 \n \
                                v_mac_f32 %15, %69, %79 \n \
                                v_mac_f32 %16, %69, %80 \n \
                                v_mac_f32 %17, %69, %81 \n \
                                v_mac_f32 %18, %69, %82 \n \
                                v_mac_f32 %19, %69, %83 \n \
                                v_mac_f32 %20, %70, %76 \n \
                                v_mac_f32 %21, %70, %77 \n \
                                v_mac_f32 %22, %70, %78 \n \
                                v_mac_f32 %23, %70, %79 \n \
                                v_mac_f32 %24, %70, %80 \n \
                                v_mac_f32 %25, %70, %81 \n \
                                v_mac_f32 %26, %70, %82 \n \
                                v_mac_f32 %27, %70, %83 \n \
                                v_mac_f32 %28, %71, %76 \n \
                                v_mac_f32 %29, %71, %77 \n \
                                v_mac_f32 %30, %71, %78 \n \
                                v_mac_f32 %31, %71, %79 \n \
                                v_mac_f32 %32, %71, %80 \n \
                                v_mac_f32 %33, %71, %81 \n \
                                v_mac_f32 %34, %71, %82 \n \
                                v_mac_f32 %35, %71, %83 \n \
                                v_mac_f32 %36, %72, %76 \n \
                                v_mac_f32 %37, %72, %77 \n \
                                v_mac_f32 %38, %72, %78 \n \
                                v_mac_f32 %39, %72, %79 \n \
                                v_mac_f32 %40, %72, %80 \n \
                                v_mac_f32 %41, %72, %81 \n \
                                v_mac_f32 %42, %72, %82 \n \
                                v_mac_f32 %43, %72, %83 \n \
                                v_mac_f32 %44, %73, %76 \n \
                                v_mac_f32 %45, %73, %77 \n \
                                v_mac_f32 %46, %73, %78 \n \
                                v_mac_f32 %47, %73, %79 \n \
                                v_mac_f32 %48, %73, %80 \n \
                                v_mac_f32 %49, %73, %81 \n \
                                v_mac_f32 %50, %73, %82 \n \
                                v_mac_f32 %51, %73, %83 \n \
                                v_mac_f32 %52, %74, %76 \n \
                                v_mac_f32 %53, %74, %77 \n \
                                v_mac_f32 %54, %74, %78 \n \
                                v_mac_f32 %55, %74, %79 \n \
                                v_mac_f32 %56, %74, %80 \n \
                                v_mac_f32 %57, %74, %81 \n \
                                v_mac_f32 %58, %74, %82 \n \
                                v_mac_f32 %59, %74, %83 \n \
                                v_mac_f32 %60, %75, %76 \n \
                                v_mac_f32 %61, %75, %77 \n \
                                v_mac_f32 %62, %75, %78 \n \
                                v_mac_f32 %63, %75, %79 \n \
                                v_mac_f32 %64, %75, %80 \n \
                                v_mac_f32 %65, %75, %81 \n \
                                v_mac_f32 %66, %75, %82 \n \
                                v_mac_f32 %67, %75, %83 \n \
                                "
                         : "=v"(reg[0]),
                           "=v"(reg[1]),
                           "=v"(reg[2]),
                           "=v"(reg[3]),
                           "=v"(p_c_thread[0]),
                           "=v"(p_c_thread[1]),
                           "=v"(p_c_thread[2]),
                           "=v"(p_c_thread[3]),
                           "=v"(p_c_thread[4]),
                           "=v"(p_c_thread[5]),
                           "=v"(p_c_thread[6]),
                           "=v"(p_c_thread[7]),
                           "=v"(p_c_thread[8]),
                           "=v"(p_c_thread[9]),
                           "=v"(p_c_thread[10]),
                           "=v"(p_c_thread[11]),
                           "=v"(p_c_thread[12]),
                           "=v"(p_c_thread[13]),
                           "=v"(p_c_thread[14]),
                           "=v"(p_c_thread[15]),
                           "=v"(p_c_thread[16]),
                           "=v"(p_c_thread[17]),
                           "=v"(p_c_thread[18]),
                           "=v"(p_c_thread[19]),
                           "=v"(p_c_thread[20]),
                           "=v"(p_c_thread[21]),
                           "=v"(p_c_thread[22]),
                           "=v"(p_c_thread[23]),
                           "=v"(p_c_thread[24]),
                           "=v"(p_c_thread[25]),
                           "=v"(p_c_thread[26]),
                           "=v"(p_c_thread[27]),
                           "=v"(p_c_thread[28]),
                           "=v"(p_c_thread[29]),
                           "=v"(p_c_thread[30]),
                           "=v"(p_c_thread[31]),
                           "=v"(p_c_thread[32]),
                           "=v"(p_c_thread[33]),
                           "=v"(p_c_thread[34]),
                           "=v"(p_c_thread[35]),
                           "=v"(p_c_thread[36]),
                           "=v"(p_c_thread[37]),
                           "=v"(p_c_thread[38]),
                           "=v"(p_c_thread[39]),
                           "=v"(p_c_thread[40]),
                           "=v"(p_c_thread[41]),
                           "=v"(p_c_thread[42]),
                           "=v"(p_c_thread[43]),
                           "=v"(p_c_thread[44]),
                           "=v"(p_c_thread[45]),
                           "=v"(p_c_thread[46]),
                           "=v"(p_c_thread[47]),
                           "=v"(p_c_thread[48]),
                           "=v"(p_c_thread[49]),
                           "=v"(p_c_thread[50]),
                           "=v"(p_c_thread[51]),
                           "=v"(p_c_thread[52]),
                           "=v"(p_c_thread[53]),
                           "=v"(p_c_thread[54]),
                           "=v"(p_c_thread[55]),
                           "=v"(p_c_thread[56]),
                           "=v"(p_c_thread[57]),
                           "=v"(p_c_thread[58]),
                           "=v"(p_c_thread[59]),
                           "=v"(p_c_thread[60]),
                           "=v"(p_c_thread[61]),
                           "=v"(p_c_thread[62]),
                           "=v"(p_c_thread[63])
                         : "v"(p_a_thread[0]),
                           "v"(p_a_thread[1]),
                           "v"(p_a_thread[2]),
                           "v"(p_a_thread[3]),
                           "v"(p_a_thread[4]),
                           "v"(p_a_thread[5]),
                           "v"(p_a_thread[6]),
                           "v"(p_a_thread[7]),
                           "v"(p_b_thread[0]),
                           "v"(p_b_thread[1]),
                           "v"(p_b_thread[2]),
                           "v"(p_b_thread[3]),
                           "v"(p_b_thread[4]),
                           "v"(p_b_thread[5]),
                           "v"(p_b_thread[6]),
                           "v"(p_b_thread[7]),
                           "v"(__to_local((void*)(a_loc))),
                           "v"(__to_local((void*)(b_loc))),
                           "4"(p_c_thread[0]),
                           "5"(p_c_thread[1]),
                           "6"(p_c_thread[2]),
                           "7"(p_c_thread[3]),
                           "8"(p_c_thread[4]),
                           "9"(p_c_thread[5]),
                           "10"(p_c_thread[6]),
                           "11"(p_c_thread[7]),
                           "12"(p_c_thread[8]),
                           "13"(p_c_thread[9]),
                           "14"(p_c_thread[10]),
                           "15"(p_c_thread[11]),
                           "16"(p_c_thread[12]),
                           "17"(p_c_thread[13]),
                           "18"(p_c_thread[14]),
                           "19"(p_c_thread[15]),
                           "20"(p_c_thread[16]),
                           "21"(p_c_thread[17]),
                           "22"(p_c_thread[18]),
                           "23"(p_c_thread[19]),
                           "24"(p_c_thread[20]),
                           "25"(p_c_thread[21]),
                           "26"(p_c_thread[22]),
                           "27"(p_c_thread[23]),
                           "28"(p_c_thread[24]),
                           "29"(p_c_thread[25]),
                           "30"(p_c_thread[26]),
                           "31"(p_c_thread[27]),
                           "32"(p_c_thread[28]),
                           "33"(p_c_thread[29]),
                           "34"(p_c_thread[30]),
                           "35"(p_c_thread[31]),
                           "36"(p_c_thread[32]),
                           "37"(p_c_thread[33]),
                           "38"(p_c_thread[34]),
                           "39"(p_c_thread[35]),
                           "40"(p_c_thread[36]),
                           "41"(p_c_thread[37]),
                           "42"(p_c_thread[38]),
                           "43"(p_c_thread[39]),
                           "44"(p_c_thread[40]),
                           "45"(p_c_thread[41]),
                           "46"(p_c_thread[42]),
                           "47"(p_c_thread[43]),
                           "48"(p_c_thread[44]),
                           "49"(p_c_thread[45]),
                           "50"(p_c_thread[46]),
                           "51"(p_c_thread[47]),
                           "52"(p_c_thread[48]),
                           "53"(p_c_thread[49]),
                           "54"(p_c_thread[50]),
                           "55"(p_c_thread[51]),
                           "56"(p_c_thread[52]),
                           "57"(p_c_thread[53]),
                           "58"(p_c_thread[54]),
                           "59"(p_c_thread[55]),
                           "60"(p_c_thread[56]),
                           "61"(p_c_thread[57]),
                           "62"(p_c_thread[58]),
                           "63"(p_c_thread[59]),
                           "64"(p_c_thread[60]),
                           "65"(p_c_thread[61]),
                           "66"(p_c_thread[62]),
                           "67"(p_c_thread[63]));
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
