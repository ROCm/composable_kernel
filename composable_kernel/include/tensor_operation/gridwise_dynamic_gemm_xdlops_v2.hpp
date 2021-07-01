#ifndef CK_GRIDWISE_DYNAMIC_GEMM_XDLOPS_V2_HPP
#define CK_GRIDWISE_DYNAMIC_GEMM_XDLOPS_V2_HPP

#include "common_header.hpp"
#include "dynamic_multi_index_transform_helper.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "blockwise_gemm_xdlops.hpp"
#include "blockwise_dynamic_tensor_slice_transfer.hpp"
#include "threadwise_dynamic_tensor_slice_transfer.hpp"
#include "threadwise_dynamic_tensor_slice_set.hpp"

namespace ck {

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
template <typename GridwiseGemm,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AGlobalDesc,
          typename BGlobalDesc,
          typename CGlobalDesc,
          typename CBlockClusterDesc>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_dynamic_gemm_xdlops_v2(const FloatA* __restrict__ p_a_global,
                                      const FloatB* __restrict__ p_b_global,
                                      FloatC* __restrict__ p_c_global,
                                      const AGlobalDesc a_k0_m_k1_global_desc,
                                      const BGlobalDesc b_k0_n_k1_global_desc,
                                      const CGlobalDesc c_m0_m1_m2_n_global_desc,
                                      const CBlockClusterDesc c_block_cluster_desc)
{
    GridwiseGemm::Run(p_a_global,
                      p_b_global,
                      p_c_global,
                      a_k0_m_k1_global_desc,
                      b_k0_n_k1_global_desc,
                      c_m0_m1_m2_n_global_desc,
                      c_block_cluster_desc);
}
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
// pass tensor descriptor by __CONSTANT__ void pointer
// __CONSTANT__ is needed to inform compiler void pointers in the kernel signature are pointing to
// non-modifiable parameter address space, so compiler can enable corresponding optimization
template <typename GridwiseGemm,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AGlobalDesc,
          typename BGlobalDesc,
          typename CGlobalDesc,
          typename CBlockClusterDesc>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_dynamic_gemm_xdlops_v2(const FloatA* __restrict__ p_a_global,
                                      const FloatB* __restrict__ p_b_global,
                                      FloatC* __restrict__ p_c_global,
                                      const void __CONSTANT__* p_a_k0_m_k1_global_desc,
                                      const void __CONSTANT__* p_b_k0_n_k1_global_desc,
                                      const void __CONSTANT__* p_c_m0_m1_m2_n_global_desc,
                                      const void __CONSTANT__* p_c_block_cluster_desc)
{
    // first cast void __CONSTANT__ void* to void*
    // second cast void* to Desc*
    // the copy constructor of tensor descriptor doesn't take address_space(4)
    const auto a_k0_m_k1_global_desc =
        *reinterpret_cast<const AGlobalDesc*>((const void*)p_a_k0_m_k1_global_desc);
    const auto b_k0_n_k1_global_desc =
        *reinterpret_cast<const BGlobalDesc*>((const void*)p_b_k0_n_k1_global_desc);
    const auto c_m0_m1_m2_n_global_desc =
        *reinterpret_cast<const CGlobalDesc*>((const void*)p_c_m0_m1_m2_n_global_desc);

    const auto c_block_cluster_desc =
        *reinterpret_cast<const CBlockClusterDesc*>((const void*)p_c_block_cluster_desc);

    GridwiseGemm::Run(p_a_global,
                      p_b_global,
                      p_c_global,
                      a_k0_m_k1_global_desc,
                      b_k0_n_k1_global_desc,
                      c_m0_m1_m2_n_global_desc,
                      c_block_cluster_desc,
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
}
#endif

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AGlobalDesc,
          typename BGlobalDesc,
          typename CGlobalDesc,
          typename CBlockClusterDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPack,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadSliceLengths_K_M_KPack,
          typename ABlockTransferThreadClusterLengths_K_M_KPack,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_KPack,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K_N_KPack,
          typename BBlockTransferThreadClusterLengths_K_N_KPack,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_KPack,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGlobalIteratorHacks,
          typename BGlobalIteratorHacks,
          typename CGlobalIteratorHacks,
          typename AGlobalMoveSliceWindowIteratorHacks,
          typename BGlobalMoveSliceWindowIteratorHacks>
struct GridwiseDynamicGemm_km_kn_m0m1n0n1_xdlops_v2
{
    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto max_lds_align = Number<KPack>{};

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k0_m_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<MPerBlock>{}, Number<KPack>{}), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k0_n_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<NPerBlock>{}, Number<KPack>{}), max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k0_m_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size =
            math::integer_least_multiple(b_k0_n_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        return (a_block_space_size + b_block_space_size) * sizeof(FloatAB);
    }

    __device__ static void Run(const FloatAB* __restrict__ p_a_global,
                               const FloatAB* __restrict__ p_b_global,
                               FloatC* __restrict__ p_c_global,
                               const AGlobalDesc& a_k0_m_k1_global_desc,
                               const BGlobalDesc& b_k0_n_k1_global_desc,
                               const CGlobalDesc& c_m0_m1_m2_n_global_desc,
                               const CBlockClusterDesc& c_block_cluster_desc,
                               FloatAB* __restrict__ p_shared_block)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        const auto a_global_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_a_global, a_k0_m_k1_global_desc.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_b_global, b_k0_n_k1_global_desc.GetElementSpaceSize());
        auto c_global_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_c_global, c_m0_m1_m2_n_global_desc.GetElementSpaceSize());

        const auto K0 = a_k0_m_k1_global_desc.GetLength(I0);
        const auto M  = a_k0_m_k1_global_desc.GetLength(I1);
        const auto N  = b_k0_n_k1_global_desc.GetLength(I1);
        const auto K1 = b_k0_n_k1_global_desc.GetLength(I2);

        // divide block work by [M, N]
        const auto block_work_idx =
            c_block_cluster_desc.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        // HACK: this force m/n_block_data_idx_on_global into SGPR
        const index_t m_block_data_idx_on_global =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_global =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = Number<KPack>{};

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k0_m_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<MPerBlock>{}, Number<KPack>{}), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k0_n_k1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<NPerBlock>{}, Number<KPack>{}), max_lds_align);

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<KPerBlock, MPerBlock, KPack>,
                                                   ABlockTransferThreadSliceLengths_K_M_KPack,
                                                   ABlockTransferThreadClusterLengths_K_M_KPack,
                                                   ABlockTransferThreadClusterArrangeOrder,
                                                   FloatAB,
                                                   FloatAB,
                                                   decltype(a_k0_m_k1_global_desc),
                                                   decltype(a_k0_m_k1_block_desc),
                                                   ABlockTransferSrcAccessOrder,
                                                   Sequence<1, 0, 2>,
                                                   ABlockTransferSrcVectorDim,
                                                   2,
                                                   ABlockTransferSrcScalarPerVector,
                                                   ABlockTransferDstScalarPerVector_KPack,
                                                   1,
                                                   1,
                                                   AThreadTransferSrcResetCoordinateAfterRun,
                                                   true>(
                a_k0_m_k1_global_desc,
                make_multi_index(0, m_block_data_idx_on_global, 0),
                a_k0_m_k1_block_desc,
                make_multi_index(0, 0, 0));

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<KPerBlock, NPerBlock, KPack>,
                                                   BBlockTransferThreadSliceLengths_K_N_KPack,
                                                   BBlockTransferThreadClusterLengths_K_N_KPack,
                                                   BBlockTransferThreadClusterArrangeOrder,
                                                   FloatAB,
                                                   FloatAB,
                                                   decltype(b_k0_n_k1_global_desc),
                                                   decltype(b_k0_n_k1_block_desc),
                                                   BBlockTransferSrcAccessOrder,
                                                   Sequence<1, 0, 2>,
                                                   BBlockTransferSrcVectorDim,
                                                   2,
                                                   BBlockTransferSrcScalarPerVector,
                                                   BBlockTransferDstScalarPerVector_KPack,
                                                   1,
                                                   1,
                                                   BThreadTransferSrcResetCoordinateAfterRun,
                                                   true>(
                b_k0_n_k1_global_desc,
                make_multi_index(0, n_block_data_idx_on_global, 0),
                b_k0_n_k1_block_desc,
                make_multi_index(0, 0, 0));

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check

        static_assert(MPerBlock % (MPerWave * MRepeat) == 0 &&
                          NPerBlock % (NPerWave * NRepeat) == 0,
                      "wrong!");

        constexpr auto a_k0_m0_m1_k1_block_desc = transform_dynamic_tensor_descriptor(
            a_k0_m_k1_block_desc,
            make_tuple(make_pass_through_transform(Number<KPerBlock>{}),
                       make_unmerge_transform(
                           make_tuple(Number<MRepeat>{}, Number<MPerBlock / MRepeat>{})),
                       make_pass_through_transform(Number<KPack>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        constexpr auto b_k0_n0_n1_k1_block_desc = transform_dynamic_tensor_descriptor(
            b_k0_n_k1_block_desc,
            make_tuple(make_pass_through_transform(Number<KPerBlock>{}),
                       make_unmerge_transform(
                           make_tuple(Number<NRepeat>{}, Number<NPerBlock / NRepeat>{})),
                       make_pass_through_transform(Number<KPack>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        const auto blockwise_gemm =
            BlockwiseGemmXdlops_km_kn_m0m1m2n_v1<BlockSize,
                                                 FloatAB,
                                                 decltype(a_k0_m0_m1_k1_block_desc),
                                                 decltype(b_k0_n0_n1_k1_block_desc),
                                                 MPerWave,
                                                 NPerWave,
                                                 KPack>{};

        constexpr auto CLayout = blockwise_gemm.GetCLayout();

        constexpr index_t BlkSize   = CLayout.GetBlkSize();
        constexpr index_t NumBlks   = CLayout.GetNumBlks();
        constexpr index_t NumXdlops = CLayout.GetNumXdlops();

        constexpr auto c_mr_nr_nx_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, Number<NumXdlops>{}));

        constexpr auto c_blk_nb_bs_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<NumBlks>{}, Number<BlkSize>{}));

        StaticBuffer<AddressSpace::Vgpr,
                     vector_type<FloatAcc, c_blk_nb_bs_desc.GetElementSpaceSize()>,
                     c_mr_nr_nx_desc.GetElementSpaceSize()>
            c_thread_buf;

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k0_m_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size =
            math::integer_least_multiple(b_k0_n_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        FloatAB* p_a_block = p_shared_block;
        FloatAB* p_b_block = p_shared_block + a_block_space_size;

        // register allocation for output
        // auto c_thread_buf = make_static_buffer<AddressSpace::Vgpr, FloatAcc>(
        // c_m0_m1_n0_n1_thread_desc.GetElementSpaceSize());

        // ThreadwiseDynamicTensorSliceSet_v1<FloatAcc,
        // decltype(c_m0_m1_n0_n1_thread_desc),
        // Sequence<MRepeat, MPerThread, NRepeat, NPerThread>>{}
        //.Run(c_m0_m1_n0_n1_thread_desc, make_tuple(I0, I0, I0, I0), c_thread_buf, FloatAcc{0});

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0);

        // hack to control index calculation when iterating over A and B matrix for threadwise copy
        constexpr auto a_k0_m_k1_global_iterator_hacks = AGlobalIteratorHacks{};
        constexpr auto b_k0_n_k1_global_iterator_hacks = BGlobalIteratorHacks{};

        // hack to control index calculation when move slice window for A and B matrix for
        // threadwise copy
        constexpr auto a_k0_m_k1_global_move_slice_window_iterator_hack =
            AGlobalMoveSliceWindowIteratorHacks{};
        constexpr auto b_k0_n_k1_global_move_slice_window_iterator_hack =
            BGlobalMoveSliceWindowIteratorHacks{};

        auto a_block_buf = make_dynamic_buffer<AddressSpace::Lds>(
            p_a_block, a_k0_m_k1_block_desc.GetElementSpaceSize());
        auto b_block_buf = make_dynamic_buffer<AddressSpace::Lds>(
            p_b_block, b_k0_n_k1_block_desc.GetElementSpaceSize());

        // preload data into LDS
        {
            a_blockwise_copy.RunRead(
                a_k0_m_k1_global_desc, a_global_buf, a_k0_m_k1_global_iterator_hacks);
            b_blockwise_copy.RunRead(
                b_k0_n_k1_global_desc, b_global_buf, b_k0_n_k1_global_iterator_hacks);

            a_blockwise_copy.RunWrite(a_k0_m_k1_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_k0_n_k1_block_desc, b_block_buf);
        }

        // main body
        index_t k_block_data_begin = 0;

        do
        {
            a_blockwise_copy.MoveSrcSliceWindow(a_k0_m_k1_global_desc,
                                                a_block_slice_copy_step,
                                                a_k0_m_k1_global_move_slice_window_iterator_hack);
            b_blockwise_copy.MoveSrcSliceWindow(b_k0_n_k1_global_desc,
                                                b_block_slice_copy_step,
                                                b_k0_n_k1_global_move_slice_window_iterator_hack);

            a_blockwise_copy.RunRead(
                a_k0_m_k1_global_desc, a_global_buf, a_k0_m_k1_global_iterator_hacks);
            b_blockwise_copy.RunRead(
                b_k0_n_k1_global_desc, b_global_buf, b_k0_n_k1_global_iterator_hacks);

            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();

            a_blockwise_copy.RunWrite(a_k0_m_k1_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_k0_n_k1_block_desc, b_block_buf);

            k_block_data_begin += KPerBlock;
        } while(k_block_data_begin < (K0 - KPerBlock));

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }

        // output: register to global memory
        {

            constexpr index_t M0 = CLayout.M1();
            constexpr index_t M1 = CLayout.N1();
            constexpr index_t M2 = CLayout.M0();

            constexpr auto c_m0_m1_m2_n_thread_desc =
                make_dynamic_naive_tensor_descriptor_packed_v2(
                    make_tuple(Number<M0>{}, Number<1>{}, Number<M2>{}, Number<1>{}));

            StaticBuffer<AddressSpace::Vgpr, FloatC, BlkSize> c_blk_buf_;

            static_for<0, MRepeat, 1>{}([&](auto mr_i) {
                static_for<0, NRepeat, 1>{}([&](auto nr_i) {
                    static_for<0, NumXdlops, 1>{}([&](auto xdlops_i) {
                        static_for<0, NumBlks, 1>{}([&](auto blk_i) {
                            auto c_blk = c_thread_buf[Number<c_mr_nr_nx_desc.CalculateOffset(
                                make_tuple(mr_i, nr_i, xdlops_i))>{}];

                            static_for<0, BlkSize, 1>{}([&](auto j) {
                                c_blk_buf_(j) = c_blk.template AsType<FloatAcc>()[Number<
                                    c_blk_nb_bs_desc.CalculateOffset(make_tuple(blk_i, j))>{}];
                            });

                            // calculate origin of thread output tensor on global memory
                            //     blockwise GEMM c matrix starting index
                            const auto c_thread_mtx_on_block =
                                blockwise_gemm.CalculateCThreadOriginDataIndex(
                                    mr_i, nr_i, xdlops_i, blk_i);

                            const index_t m_thread_data_on_global =
                                m_block_data_idx_on_global + c_thread_mtx_on_block[I0];

                            const index_t n_thread_data_on_global =
                                n_block_data_idx_on_global + c_thread_mtx_on_block[I1];

                            constexpr auto c_m0_m1_m2_n_global_tensor_iterator_hacks =
                                CGlobalIteratorHacks{};

                            ThreadwiseDynamicTensorSliceTransfer_v1r3<
                                FloatC,
                                FloatC,
                                decltype(c_m0_m1_m2_n_thread_desc),
                                decltype(c_m0_m1_m2_n_global_desc),
                                Sequence<M0, 1, M2, 1>,
                                CThreadTransferSrcDstAccessOrder,
                                CThreadTransferSrcDstVectorDim,
                                CThreadTransferDstScalarPerVector,
                                CGlobalMemoryDataOperation,
                                1,
                                true>{c_m0_m1_m2_n_global_desc,
                                      make_multi_index(m_thread_data_on_global / (M2 * M1),
                                                       m_thread_data_on_global % (M2 * M1) / M2,
                                                       m_thread_data_on_global % M2,
                                                       n_thread_data_on_global)}
                                .Run(c_m0_m1_m2_n_thread_desc,
                                     make_tuple(I0, I0, I0, I0),
                                     c_blk_buf_,
                                     c_m0_m1_m2_n_global_desc,
                                     c_global_buf,
                                     c_m0_m1_m2_n_global_tensor_iterator_hacks);
                        });
                    });
                });
            });
        }
    }

    __device__ static void Run(const FloatAB* __restrict__ p_a_global,
                               const FloatAB* __restrict__ p_b_global,
                               FloatC* __restrict__ p_c_global,
                               const AGlobalDesc& a_k0_m_k1_global_desc,
                               const BGlobalDesc& b_k0_n_k1_global_desc,
                               const CGlobalDesc& c_m0_m1_m2_n_global_desc,
                               const CBlockClusterDesc& c_block_cluster_desc)
    {
        constexpr index_t shared_block_size = GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

        __shared__ FloatAB p_shared_block[shared_block_size];

        Run(p_a_global,
            p_b_global,
            p_c_global,
            a_k0_m_k1_global_desc,
            b_k0_n_k1_global_desc,
            c_m0_m1_m2_n_global_desc,
            c_block_cluster_desc,
            p_shared_block);
    }
};

} // namespace ck
#endif
