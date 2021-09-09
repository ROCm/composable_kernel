#ifndef CK_GRIDWISE_GEMM_V2_HPP
#define CK_GRIDWISE_GEMM_V2_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_tensor_slice_transfer.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "blockwise_gemm_dlops_v3.hpp"

namespace ck {

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_E0_E1_K,
          typename BGridDesc_E_N_Ho_Wo,
          typename CGridDesc_K_N_Ho_Wo,
          typename CBlockIdToBlockClusterAdaptor_K_N_Ho_Wo,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v2(const FloatAB* __restrict__ p_a_grid,
                             const FloatAB* __restrict__ p_b_grid,
                             FloatC* __restrict__ p_c_grid,
                             const AGridDesc_E0_E1_K a_e0_e1_k_grid_desc,
                             const BGridDesc_E_N_Ho_Wo b_e0_e1_n_ho_wo_grid_desc,
                             const CGridDesc_K_N_Ho_Wo c_k_n_ho_wo_grid_desc,
                             const CBlockIdToBlockClusterAdaptor_K_N_Ho_Wo
                                 c_blockid_to_k_n_ho_wo_block_cluster_adaptor)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_shared_block,
                      a_e0_e1_k_grid_desc,
                      b_e0_e1_n_ho_wo_grid_desc,
                      c_k_n_ho_wo_grid_desc,
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
}
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
// pass tensor descriptor by CONSTANT void pointer
// CONSTANT is needed to inform compiler void pointers in the kernel signature are pointing to
// non-modifiable parameter address space, so compiler can enable corresponding optimization
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_E0_E1_K,
          typename BGridDesc_E_N_Ho_Wo,
          typename CGridDesc_K_N_Ho_Wo,
          typename CBlockIdToBlockClusterAdaptor_K_N_Ho_Wo,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v2(const FloatAB* __restrict__ p_a_grid,
                             const FloatAB* __restrict__ p_b_grid,
                             FloatC* __restrict__ p_c_grid,
                             const void CONSTANT* p_a_e0_e1_k_grid_desc,
                             const void CONSTANT* p_b_e0_e1_n_ho_wo_grid_desc,
                             const void CONSTANT* p_c_k_n_ho_wo_grid_desc,
                             const void CONSTANT* p_c_blockid_to_k_n_ho_wo_block_cluster_adaptor)
{
    // first cast void CONSTANT void* to void*
    // second cast void* to Desc*
    // the copy constructor of tensor descriptor doesn't take address_space(4)
    const auto a_e0_e1_k_grid_desc = *reinterpret_cast<const AGridDesc_E0_E1_K*>(
        cast_pointer_to_generic_address_space(p_a_e0_e1_k_grid_desc));
    const auto b_e0_e1_n_ho_wo_grid_desc = *reinterpret_cast<const BGridDesc_E_N_Ho_Wo*>(
        cast_pointer_to_generic_address_space(p_b_e0_e1_n_ho_wo_grid_desc));
    const auto c_k_n_ho_wo_grid_desc = *reinterpret_cast<const CGridDesc_K_N_Ho_Wo*>(
        cast_pointer_to_generic_address_space(p_c_k_n_ho_wo_grid_desc));

    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_shared_block,
                      a_e0_e1_k_grid_desc,
                      b_e0_e1_n_ho_wo_grid_desc,
                      c_k_n_ho_wo_grid_desc,
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
}
#endif

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum_t CGlobalMemoryDataOperation,
          typename AGlobalDesc_E0_E1_K,
          typename BGlobalDesc_E0_E1_N_Ho_Wo,
          typename CGlobalDesc_K_N_Ho_Wo,
          index_t KPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t EPerBlock,
          index_t KPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E_K,
          typename ABlockTransferThreadClusterLengths_E_K,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGlobalStepHacks,
          typename BGlobalStepHacks,
          typename CGlobalStepHacks,
          typename AGlobalMoveSliceWindowStepHacks,
          typename BGlobalMoveSliceWindowStepHacks>
struct GridwiseGemmDlops_km_kn_mn_v3
{
    static constexpr auto E = EPerBlock;

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto max_lds_align =
            math::lcm(Number<ABlockTransferDstScalarPerVector_K>{}, Number<KPerBlock>{});

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_e1_k_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E>{}, Number<KPerBlock>{}), max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_e1_k_block_desc.GetElementSpaceSize(), max_lds_align);

        return a_block_space_size * sizeof(FloatAB);
    }

    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ static void Run(const FloatAB* __restrict__ p_a_global,
                               const FloatAB* __restrict__ p_b_global,
                               FloatC* __restrict__ p_c_global,
                               FloatAB* __restrict__ p_shared_block,
                               const AGlobalDesc_E0_E1_K& a_e0_e1_k_global_desc,
                               const BGlobalDesc_E0_E1_N_Ho_Wo& b_e0_e1_n_ho_wo_global_desc,
                               const CGlobalDesc_K_N_Ho_Wo& c_k_n_ho_wo_global_desc,
                               integral_constant<bool, HasMainKBlockLoop>,
                               integral_constant<bool, HasDoubleTailKBlockLoop>)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_a_global, a_e0_e1_k_global_desc.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_b_global, b_e0_e1_n_ho_wo_global_desc.GetElementSpaceSize());
        auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_c_global, c_k_n_ho_wo_global_desc.GetElementSpaceSize());

        // const auto E = a_e0_e1_k_global_desc.GetLength(I0);
        // const auto K = a_e0_e1_k_global_desc.GetLength(I1);

        // const auto N  = b_e0_e1_n_ho_wo_global_desc.GetLength(I1);
        const auto Ho = b_e0_e1_n_ho_wo_global_desc.GetLength(I2);
        const auto Wo = b_e0_e1_n_ho_wo_global_desc.GetLength(I3);

// divide block work by [M, N]
#if 0
        const auto ho_block_work_num  = Ho / Number<HoPerBlock>{};
        const auto wo_block_work_num  = Wo / Number<WoPerBlock>{};
        const auto hwo_block_work_num = ho_block_work_num * wo_block_work_num;

        const index_t k_block_work_id   = get_block_1d_id() / hwo_block_work_num;
        const index_t hwo_block_work_id = get_block_1d_id() - k_block_work_id * hwo_block_work_num;

        const index_t ho_block_work_id = hwo_block_work_id / wo_block_work_num;
        const index_t wo_block_work_id = hwo_block_work_id - ho_block_work_id * wo_block_work_num;
#else
        // Hack: this force result into SGPR
        const index_t ho_block_work_num  = __builtin_amdgcn_readfirstlane(Ho / HoPerBlock);
        const index_t wo_block_work_num  = __builtin_amdgcn_readfirstlane(Wo / WoPerBlock);
        const index_t hwo_block_work_num = ho_block_work_num * wo_block_work_num;

        const index_t k_block_work_id =
            __builtin_amdgcn_readfirstlane(get_block_1d_id() / hwo_block_work_num);
        const index_t hwo_block_work_id = get_block_1d_id() - k_block_work_id * hwo_block_work_num;

        const index_t ho_block_work_id =
            __builtin_amdgcn_readfirstlane(hwo_block_work_id / wo_block_work_num);
        const index_t wo_block_work_id = hwo_block_work_id - ho_block_work_id * wo_block_work_num;
#endif

        // lds max alignment
        constexpr auto max_lds_align =
            math::lcm(Number<ABlockTransferDstScalarPerVector_K>{}, Number<KPerBlock>{});

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_e1_k_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E>{}, Number<KPerBlock>{}), max_lds_align);

        constexpr auto a_e2_k_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<EPerBlock>{}, Number<KPerBlock>{}), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_e2_n_ho_wo_block_desc = make_naive_tensor_descriptor_packed(make_tuple(
            Number<EPerBlock>{}, Number<1>{}, Number<HoPerBlock>{}, Number<WoPerBlock>{}));

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_k_n_ho_wo_thread_desc = make_naive_tensor_descriptor_packed(make_tuple(
            Number<KPerThread>{}, Number<1>{}, Number<HoPerThread>{}, Number<WoPerThread>{}));

        auto blockwise_gemm =
            BlockwiseGemmDlops_km_kn_m0m1n0n1_v3<BlockSize,
                                                 FloatAB,
                                                 FloatAcc,
                                                 decltype(a_e2_k_block_desc),
                                                 decltype(b_e2_n_ho_wo_block_desc),
                                                 decltype(c_k_n_ho_wo_thread_desc),
                                                 EPerThread,
                                                 ABlockTransferSrcScalarPerVector,
                                                 ABlockTransferDstScalarPerVector_K>{};

        auto c_thread_mtx_index = blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        const auto k_thread_id  = c_thread_mtx_index.k;
        const auto ho_thread_id = c_thread_mtx_index.h;
        const auto wo_thread_id = c_thread_mtx_index.w;

        const index_t k_block_data_on_global  = k_block_work_id * KPerBlock;
        const index_t ho_block_data_on_global = ho_block_work_id * HoPerBlock;
        const index_t wo_block_data_on_global = wo_block_work_id * WoPerBlock;

        const index_t ho_thread_data_on_global =
            ho_block_data_on_global + ho_thread_id * HoPerThread;
        const index_t wo_thread_data_on_global =
            wo_block_data_on_global + wo_thread_id * WoPerThread;

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4<BlockSize,
                                            InMemoryDataOperationEnum_t::Set,
                                            Sequence<E, KPerBlock>,
                                            ABlockTransferThreadSliceLengths_E_K,
                                            ABlockTransferThreadClusterLengths_E_K,
                                            ABlockTransferThreadClusterArrangeOrder,
                                            FloatAB,
                                            FloatAB,
                                            decltype(a_e0_e1_k_global_desc),
                                            decltype(a_e1_k_block_desc),
                                            ABlockTransferSrcAccessOrder,
                                            Sequence<0, 1>,
                                            ABlockTransferSrcVectorDim,
                                            1,
                                            ABlockTransferSrcScalarPerVector,
                                            ABlockTransferDstScalarPerVector_K,
                                            1,
                                            1,
                                            AThreadTransferSrcResetCoordinateAfterRun,
                                            true>(a_e0_e1_k_global_desc,
                                                  make_multi_index(0, k_block_data_on_global),
                                                  a_e1_k_block_desc,
                                                  make_multi_index(0, 0));

        constexpr auto b_e2_n_ho_wo_thread_desc = make_naive_tensor_descriptor_packed(make_tuple(
            Number<EPerBlock>{}, Number<1>{}, Number<HoPerThread>{}, Number<WoPerThread>{}));

        auto b_threadwise_transfer =
            ThreadwiseTensorSliceTransfer_v2<FloatAB,
                                             FloatAB,
                                             decltype(b_e0_e1_n_ho_wo_global_desc),
                                             decltype(b_e2_n_ho_wo_thread_desc),
                                             Sequence<EPerBlock, 1, HoPerThread, WoPerThread>,
                                             BBlockTransferSrcAccessOrder,
                                             BBlockTransferSrcVectorDim,
                                             BBlockTransferSrcScalarPerVector,
                                             1,
                                             true>(
                b_e0_e1_n_ho_wo_global_desc,
                make_multi_index(0, 0, ho_thread_data_on_global, wo_thread_data_on_global));

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            p_shared_block, a_e1_k_block_desc.GetElementSpaceSize());

        // register allocation for output
        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatAcc,
                     c_k_n_ho_wo_thread_desc.GetElementSpaceSize(),
                     true>
            c_thread_buf;

        // initialize output thread tensor
        ThreadwiseTensorSliceSet_v1<FloatAcc,
                                    decltype(c_k_n_ho_wo_thread_desc),
                                    Sequence<KPerThread, 1, HoPerThread, WoPerThread>>{}
            .Run(c_k_n_ho_wo_thread_desc, make_tuple(I0, I0, I0, I0), c_thread_buf, FloatAcc{0});

        constexpr auto b_thread_slice_copy_step = make_multi_index(EPerBlock, 0, 0, 0);

        // hack to control index calculation when iterating over A and B matrix for threadwise copy
        constexpr auto a_e0_e1_k_global_step_hacks       = AGlobalStepHacks{};
        constexpr auto b_e0_e1_n_ho_wo_global_step_hacks = BGlobalStepHacks{};

        // double regsiter buffer for b
        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatAB,
                     b_e2_n_ho_wo_thread_desc.GetElementSpaceSize(),
                     true>
            b_thread_even_buf, b_thread_odd_buf;

        // LDS double buffer: preload data
        {
            a_blockwise_copy.RunRead(
                a_e0_e1_k_global_desc, a_global_buf, a_e0_e1_k_global_step_hacks);

            b_threadwise_transfer.Run(b_e0_e1_n_ho_wo_global_desc,
                                      b_global_buf,
                                      b_e2_n_ho_wo_thread_desc,
                                      make_tuple(I0, I0, I0, I0),
                                      b_thread_even_buf,
                                      b_e0_e1_n_ho_wo_global_step_hacks);

            a_blockwise_copy.RunWrite(a_e1_k_block_desc, a_block_buf);
        }

        __syncthreads();

        if constexpr(HasMainKBlockLoop)
        {
            index_t e_block_data_begin = 0;

            // LDS double buffer: main body
            // use Do-While loop instead of For loop to simplify control flow
            do
            {
                // even iteration
                b_threadwise_transfer.MoveSrcSliceWindow(b_e0_e1_n_ho_wo_global_desc,
                                                         b_thread_slice_copy_step);

                b_threadwise_transfer.Run(b_e0_e1_n_ho_wo_global_desc,
                                          b_global_buf,
                                          b_e2_n_ho_wo_thread_desc,
                                          make_tuple(I0, I0, I0, I0),
                                          b_thread_odd_buf,
                                          b_e0_e1_n_ho_wo_global_step_hacks);

                // LDS double buffer: GEMM on current data
                // TODO: @Zhang Jing: blockwise gemm should be able to move slice window
                blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                blockwise_gemm.MoveABlockSliceWindow(make_tuple(EPerBlock, 0));

                b_threadwise_transfer.MoveSrcSliceWindow(b_e0_e1_n_ho_wo_global_desc,
                                                         b_thread_slice_copy_step);

                b_threadwise_transfer.Run(b_e0_e1_n_ho_wo_global_desc,
                                          b_global_buf,
                                          b_e2_n_ho_wo_thread_desc,
                                          make_tuple(I0, I0, I0, I0),
                                          b_thread_even_buf,
                                          b_e0_e1_n_ho_wo_global_step_hacks);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);

                blockwise_gemm.MoveABlockSliceWindow(make_tuple(EPerBlock, 0));

                e_block_data_begin += 2 * EPerBlock;

            } while(e_block_data_begin < E - 2 * EPerBlock);
        }

        // LDS double buffer: tail
        if constexpr(HasDoubleTailKBlockLoop) // if has 2 iteration left
        {
            b_threadwise_transfer.MoveSrcSliceWindow(b_e0_e1_n_ho_wo_global_desc,
                                                     b_thread_slice_copy_step);

            b_threadwise_transfer.Run(b_e0_e1_n_ho_wo_global_desc,
                                      b_global_buf,
                                      b_e2_n_ho_wo_thread_desc,
                                      make_tuple(I0, I0, I0, I0),
                                      b_thread_odd_buf,
                                      b_e0_e1_n_ho_wo_global_step_hacks);

            // LDS double buffer: GEMM on 2nd-last data
            blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

            blockwise_gemm.MoveABlockSliceWindow(make_tuple(EPerBlock, 0));

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);
        }
        else // if has 1 iteration left
        {
            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);
        }

        // output: register to global memory
        {
            // hack to control index calculation when iterating over c_k_n_ho_wo_global tensor
            constexpr auto c_k_n_ho_wo_global_tensor_step_hacks = CGlobalStepHacks{};

            const index_t k_thread_data_on_global =
                k_block_data_on_global + k_thread_id * KPerThread;

            ThreadwiseTensorSliceTransfer_v1r3<FloatAcc,
                                               FloatC,
                                               decltype(c_k_n_ho_wo_thread_desc),
                                               decltype(c_k_n_ho_wo_global_desc),
                                               Sequence<KPerThread, 1, HoPerThread, WoPerThread>,
                                               CThreadTransferSrcDstAccessOrder,
                                               CThreadTransferSrcDstVectorDim,
                                               CThreadTransferDstScalarPerVector,
                                               CGlobalMemoryDataOperation,
                                               1,
                                               true>(
                c_k_n_ho_wo_global_desc,
                make_multi_index(
                    k_thread_data_on_global, 0, ho_thread_data_on_global, wo_thread_data_on_global))
                .Run(c_k_n_ho_wo_thread_desc,
                     make_tuple(I0, I0, I0, I0),
                     c_thread_buf,
                     c_k_n_ho_wo_global_desc,
                     c_global_buf,
                     c_k_n_ho_wo_global_tensor_step_hacks);
        }
    }
};

} // namespace ck
#endif
