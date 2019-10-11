#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R3_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R3_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER

#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "ConstantMergedTensorDescriptor_deprecated.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "blockwise_gemm.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          class ConvStrides,
          class ConvDilations,
          index_t N0,
          index_t N1,
          index_t N2,
          index_t Ho0,
          index_t Ho1,
          index_t Ho2,
          index_t Wo0,
          index_t Wo1,
          index_t Wo2,
          index_t BPerBlock,
          index_t KPerBlock,
          index_t EPerBlock,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class InBlockCopySubLengths_E_N1_Ho1_Wo1_B_N2_Ho2_Wo2,
          class InBlockCopyClusterLengths_E_N1_Ho1_Wo1_B_N2_Ho2_Wo2,
          class InBlockCopyThreadClusterArrangeOrder,
          class InBlockCopySrcAccessOrder,
          class InBlockCopyDstAccessOrder,
          index_t InBlockCopyDataPerAccess_W2,
          class WeiBlockCopySubLengths_E_K,
          class WeiBlockCopyClusterLengths_E_K,
          class WeiBlockCopyThreadClusterArrangeOrder,
          class WeiBlockCopySrcAccessOrder,
          class WeiBlockCopyDstAccessOrder,
          index_t WeiBlockCopySrcDataPerRead_E,
          index_t WeiBlockCopyDstDataPerWrite_K>
struct GridwiseConvolutionImplicitGemm_v4r3_nchw_kcyx_nkhw_lds_double_buffer
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        // this is a mess
        // TODO: find more elegent way of specifying (or calculating) performance parameters
        static_assert(N2 * Ho2 * Wo2 == GemmNPerThreadSubC, "wrong!");
        static_assert((N1 * Ho1 * Wo1 * BPerBlock * N2 * Ho2 * Wo2) %
                              (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I7 = Number<7>{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto in_n_c_h_w_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_c_y_x_global_desc = WeiGlobalDesc{};
        constexpr auto out_n_k_h_w_global_desc = OutGlobalDesc{};

        constexpr index_t N = in_n_c_h_w_global_desc.GetLengths()[0];
        constexpr index_t C = in_n_c_h_w_global_desc.GetLengths()[1];

        constexpr index_t K  = out_n_k_h_w_global_desc.GetLengths()[1];
        constexpr index_t Ho = out_n_k_h_w_global_desc.GetLengths()[2];
        constexpr index_t Wo = out_n_k_h_w_global_desc.GetLengths()[3];

        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLengths()[2];
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLengths()[3];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t E = C * Y * X;

        constexpr index_t B = N0 * Ho0 * Wo0;

        static_assert(N == N0 * N1 * N2 && Ho == Ho0 * Ho1 * Ho2 && Wo == Wo0 * Wo1 * Wo2,
                      "wrong!");

        static_assert((X == 1 || ConvDilationW % InBlockCopyDataPerAccess_W2 == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // divide block work by [K, B]
        static_assert(K % KPerBlock == 0 && B % BPerBlock == 0 && E % (2 * EPerBlock) == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t KBlockWork = K / KPerBlock;
        constexpr index_t BBlockWork = B / BPerBlock;

        constexpr auto block_work_desc =
            make_ConstantTensorDescriptor_packed(Sequence<KBlockWork, BBlockWork>{});

        const auto block_work_multi_id =
            block_work_desc.GetMultiIndexFrom1dIndex(get_block_1d_id());

        const index_t k_block_data_on_global = block_work_multi_id[0] * KPerBlock;
        const index_t b_block_data_on_global = block_work_multi_id[1] * BPerBlock;

        // input tensor
        //     tensor descriptor in device memory [N0, N1, N2, Ho0, Ho1, Ho2, Wo0, Wo1, Wo2]
        constexpr auto in_n0_n1_n2_ho0_ho1_ho2_wo0_wo1_wo2_global_desc =
            in_n_c_h_w_global_desc.Extract(I0, I2, I3)
                .StridedSlice(I1, Number<Ho>{}, Number<ConvStrideH>{})
                .StridedSlice(I2, Number<Wo>{}, Number<ConvStrideW>{})
                .Fold(I2, Number<Wo1>{}, Number<Wo2>{})
                .Fold(I1, Number<Ho1>{}, Number<Ho2>{})
                .Fold(I0, Number<N1>{}, Number<N2>{});

        constexpr auto in_n1_ho1_wo1_n0_ho0_wo0_n2_ho2_wo2_global_desc =
            in_n0_n1_n2_ho0_ho1_ho2_wo0_wo1_wo2_global_desc.ReorderGivenNew2Old(
                Sequence<1, 4, 7, 0, 3, 6, 2, 5, 8>{});

        //     batch descritpor for device memory
        constexpr auto in_c_y_x_global_desc =
            in_n_c_h_w_global_desc.StridedSlice(I2, Number<Y>{}, Number<ConvDilationH>{})
                .StridedSlice(I3, Number<X>{}, Number<ConvDilationW>{})
                .Extract(Sequence<1, 2, 3>{});

        //     merged tensor descriptor in device memory [E, N1, B, N2], src of blockwise copy
        constexpr auto in_e_n1_ho1_wo1_b_n2_ho2_wo2_global_merged_desc =
            make_ConstantMergedTensorDescriptor(
                in_c_y_x_global_desc.Embed(in_n1_ho1_wo1_n0_ho0_wo0_n2_ho2_wo2_global_desc),
                Sequence<0, 1, 2>{},
                Sequence<3>{},
                Sequence<4>{},
                Sequence<5>{},
                Sequence<6, 7, 8>{},
                Sequence<9>{},
                Sequence<10>{},
                Sequence<11>{});

        //     memory layout descriptor in LDS [E, N1, B, N2], dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto in_e_n1_ho1_wo1_b_n2_ho2_wo2_block_desc =
            make_ConstantTensorDescriptor_packed(
                Sequence<EPerBlock, N1, Ho1, Wo1, BPerBlock, N2, Ho2, Wo2>{});

        // input blockwise copy
        //     slice a merged tensor, reorder and copy to a normal tensor
        //     this copy operator already has blockwise offset built-in
        auto blockwise_in_copy = BlockwiseGenericTensorSliceCopy_v1_deprecated<
            BlockSize,
            Float,
            decltype(in_e_n1_ho1_wo1_b_n2_ho2_wo2_global_merged_desc),
            decltype(in_e_n1_ho1_wo1_b_n2_ho2_wo2_block_desc),
            decltype(in_e_n1_ho1_wo1_b_n2_ho2_wo2_block_desc.GetLengths()),
            InBlockCopySubLengths_E_N1_Ho1_Wo1_B_N2_Ho2_Wo2,
            InBlockCopyClusterLengths_E_N1_Ho1_Wo1_B_N2_Ho2_Wo2,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopyDataPerAccess_W2,
            InBlockCopyDataPerAccess_W2>({0, 0, 0, 0, b_block_data_on_global, 0, 0, 0},
                                         {0, 0, 0, 0, 0, 0, 0, 0});

        // weight tensor
        //     tensor descriptor in device memory, src of blockwise copy
        constexpr auto wei_e_k_global_desc =
            wei_k_c_y_x_global_desc.Unfold(I1, I3).ReorderGivenNew2Old(Sequence<1, 0>{});

        //     tensor descriptor in LDS, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto wei_e_k_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<EPerBlock, KPerBlock>{},
            Number<math::lcm(WeiBlockCopyDstDataPerWrite_K, GemmDataPerReadA)>{});

        // operator for blockwise copy of weight into LDS
        //     slice a tensor, and copy it into another tensor
        //     this copy operator already have blockwise offset built-in
        auto blockwise_wei_copy =
            BlockwiseGenericTensorSliceCopy_v1_deprecated<BlockSize,
                                                          Float,
                                                          decltype(wei_e_k_global_desc),
                                                          decltype(wei_e_k_block_desc),
                                                          decltype(wei_e_k_block_desc.GetLengths()),
                                                          WeiBlockCopySubLengths_E_K,
                                                          WeiBlockCopyClusterLengths_E_K,
                                                          WeiBlockCopyThreadClusterArrangeOrder,
                                                          WeiBlockCopySrcAccessOrder,
                                                          WeiBlockCopyDstAccessOrder,
                                                          WeiBlockCopySrcDataPerRead_E,
                                                          WeiBlockCopyDstDataPerWrite_K>(
                {0, k_block_data_on_global}, {0, 0});

#if 0
        if(get_block_1d_id() == 0)
        {
            printf("id (%d %d), in offset: %d %d, wei offset %d %d\n",
                   get_block_1d_id(),
                   get_thread_local_1d_id(),
                   blockwise_in_copy.mThreadSrcOffset,
                   blockwise_in_copy.mThreadDstOffset,
                   blockwise_wei_copy.mThreadSrcOffset,
                   blockwise_wei_copy.mThreadDstOffset);
        }
#endif

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[EPerBlock, KPerBlock] is in LDS
        //     b_mtx[EPerBlocl, N1 * BPerBlock * N2] is in LDS
        //     c_mtx[KPerBlock, N1 * BPerBlock * N2] is distributed among threads, and saved in
        //     register
        constexpr auto a_e_k_block_mtx_desc = make_ConstantMatrixDescriptor(wei_e_k_block_desc);

        //     this check is ad-hoc
        //     TODO: need to properly implement tensor descriptor with multiple alignment
        //     requirements
        static_assert(in_e_n1_ho1_wo1_b_n2_ho2_wo2_block_desc.GetStrides()[3] % GemmDataPerReadB ==
                          0,
                      "GemmDataPerReadB alignment requirement is not satisfied");

        constexpr auto b_e_n1ho1wo1bn2ho2wo2_block_mtx_desc =
            make_ConstantMatrixDescriptor(in_e_n1_ho1_wo1_b_n2_ho2_wo2_block_desc.Unfold(I1, I7));

        // sanity check
        static_assert(KPerBlock % (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr index_t GemmMRepeat =
            KPerBlock / (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_k0k2_n1ho1wo1n2ho2wo2_thread_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<GemmMRepeat * GemmMPerThreadSubC>{},
                                                 Number<N1 * Ho1 * Wo1 * N2 * Ho2 * Wo2>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
            BlockSize,
            decltype(a_e_k_block_mtx_desc),
            decltype(b_e_n1ho1wo1bn2ho2wo2_block_mtx_desc),
            decltype(c_k0k2_n1ho1wo1n2ho2wo2_thread_mtx_desc),
            GemmMPerThreadSubC,
            GemmNPerThreadSubC,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmKPerThreadLoop,
            GemmDataPerReadA,
            GemmDataPerReadB>{};

        // LDS allocation for input and weight: be careful of alignment
        constexpr index_t max_align = math::lcm(InBlockCopyDataPerAccess_W2,
                                                WeiBlockCopyDstDataPerWrite_K,
                                                GemmDataPerReadA,
                                                GemmDataPerReadB);

        constexpr index_t in_block_space = math::integer_least_multiple(
            in_e_n1_ho1_wo1_b_n2_ho2_wo2_block_desc.GetElementSpace(), max_align);

        constexpr index_t wei_block_space =
            math::integer_least_multiple(wei_e_k_block_desc.GetElementSpace(), max_align);

        __shared__ Float p_in_block_double[2 * in_block_space];
        __shared__ Float p_wei_block_double[2 * wei_block_space];

        // register allocation for output
        Float p_out_thread[c_k0k2_n1ho1wo1n2ho2wo2_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_k0k2_n1ho1wo1n2ho2wo2_thread_mtx_desc, p_out_thread);

        const Float* p_wei_block_on_global = p_wei_global;

        // LDS double buffer: preload data into LDS
        {
            blockwise_in_copy.Run(p_in_global, p_in_block_double);
            blockwise_wei_copy.Run(p_wei_global, p_wei_block_double);
        }

        // LDS double buffer: main body
        for(index_t e_block_data_begin = 0; e_block_data_begin + 2 * EPerBlock < E;
            e_block_data_begin += 2 * EPerBlock)
        {
#pragma unroll
            for(index_t iloop = 0; iloop < 2; ++iloop)
            {
                const bool even_loop = (iloop % 2 == 0);

                Float* p_in_block_now =
                    even_loop ? p_in_block_double : p_in_block_double + in_block_space;
                Float* p_wei_block_now =
                    even_loop ? p_wei_block_double : p_wei_block_double + wei_block_space;

                Float* p_in_block_next =
                    even_loop ? p_in_block_double + in_block_space : p_in_block_double;
                Float* p_wei_block_next =
                    even_loop ? p_wei_block_double + wei_block_space : p_wei_block_double;

                Float p_in_register_buffer[blockwise_in_copy.GetRegisterBufferSize()];
                Float p_wei_register_buffer[blockwise_wei_copy.GetRegisterBufferSize()];

                blockwise_in_copy.MoveSlicingWindowOnSourceTensor(I0, Number<EPerBlock>{}, True);
                p_wei_block_on_global += EPerBlock * wei_e_k_global_desc.GetStride(I0);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                blockwise_in_copy.RunLoadRegisterBuffer(p_in_global, p_in_register_buffer);
                blockwise_wei_copy.RunLoadRegisterBuffer(p_wei_block_on_global,
                                                         p_wei_register_buffer);

#if 0
                if(get_block_1d_id() == 0)
                {
                    printf("tid (%d %d), %f %f %f %f\n",
                           get_block_1d_id(),
                           get_thread_local_1d_id(),
                           p_wei_register_buffer[0],
                           p_wei_register_buffer[1],
                           p_wei_register_buffer[2],
                           p_wei_register_buffer[3]);
                }
#endif

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_wei_block_now, p_in_block_now, p_out_thread);

                // LDS double buffer: store next data to LDS
                blockwise_in_copy.RunStoreRegisterBuffer(p_in_register_buffer, p_in_block_next);
                blockwise_wei_copy.RunStoreRegisterBuffer(p_wei_register_buffer, p_wei_block_next);
            }
        }

        // LDS double buffer: tail
        {
            Float p_in_register_buffer[blockwise_in_copy.GetRegisterBufferSize()];
            Float p_wei_register_buffer[blockwise_wei_copy.GetRegisterBufferSize()];

            // even iteration
            blockwise_in_copy.MoveSlicingWindowOnSourceTensor(I0, Number<EPerBlock>{}, True);
            p_wei_block_on_global += EPerBlock * wei_e_k_global_desc.GetStride(I0);

            __syncthreads();

            // LDS doubel buffer: load next data from device mem
            blockwise_in_copy.RunLoadRegisterBuffer(p_in_global, p_in_register_buffer);
            blockwise_wei_copy.RunLoadRegisterBuffer(p_wei_block_on_global, p_wei_register_buffer);

            // LDS double buffer: GEMM on current data
            blockwise_gemm.Run(p_wei_block_double, p_in_block_double, p_out_thread);

            // LDS double buffer: store next data to LDS
            blockwise_in_copy.RunStoreRegisterBuffer(p_in_register_buffer,
                                                     p_in_block_double + in_block_space);
            blockwise_wei_copy.RunStoreRegisterBuffer(p_wei_register_buffer,
                                                      p_wei_block_double + wei_block_space);

            // odd iteration
            __syncthreads();

            // LDS double buffer: GEMM on current data
            blockwise_gemm.Run(p_wei_block_double + wei_block_space,
                               p_in_block_double + in_block_space,
                               p_out_thread);
        }

        // copy output: register to global memory
        {
            constexpr index_t K2 = GemmMPerThreadSubC;
            constexpr index_t K1 = GemmMLevel0Cluster * GemmMLevel1Cluster;

            // define tensor descriptor for threadwise copy
            //     output memory layout descriptor in register
            constexpr auto out_k0_k1_k2_n1_ho1_wo1_n0_ho0_wo0_n2_ho2_wo2_thread_mem_desc =
                make_ConstantTensorDescriptor_packed(
                    Sequence<KPerBlock / (K1 * K2), 1, K2, N1, Ho1, Wo1, 1, 1, 1, N2, Ho2, Wo2>{});

            //     output tensor descriptor in register, src of threadwise copy
            constexpr auto out_n0_n1_n2_k0_k1_k2_ho0_ho1_ho2_wo0_wo1_wo2_thread_desc =
                out_k0_k1_k2_n1_ho1_wo1_n0_ho0_wo0_n2_ho2_wo2_thread_mem_desc.ReorderGivenNew2Old(
                    Sequence<6, 3, 9, 0, 1, 2, 7, 4, 10, 8, 5, 11>{});

            //     output memory layout descriptor in device memory, dst of threadwise copy
            constexpr auto out_n0_n1_n2_k0_k1_k2_ho0_ho1_ho2_wo0_wo1_wo2_global_mem_desc =
                out_n_k_h_w_global_desc.Fold(I3, Sequence<Wo1, Wo2>{})
                    .Fold(I2, Sequence<Ho1, Ho2>{})
                    .Fold(I1, Sequence<K1, K2>{})
                    .Fold(I0, Sequence<N1, N2>{});

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t k_thread_data_on_global =
                k_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t b_thread_data_on_global =
                b_block_data_on_global + c_thread_mtx_on_block.col / (N2 * Ho2 * Wo2);

            //     output merged global tensor descriptor, for calculating origin of thread tensor
            //     in global memory
            constexpr auto out_k_n1_ho1_wo1_b_n2_ho2_wo2_global_merged_desc =
                make_ConstantMergedTensorDescriptor(
                    out_n0_n1_n2_k0_k1_k2_ho0_ho1_ho2_wo0_wo1_wo2_global_mem_desc.Unfold(I3, I5),
                    Sequence<3>{},
                    Sequence<1>{},
                    Sequence<5>{},
                    Sequence<8>{},
                    Sequence<0, 4, 7>{},
                    Sequence<2>{},
                    Sequence<6>{},
                    Sequence<9>{});

            //     origin of dst in device memory
            Float* p_out_thread_on_global =
                p_out_global +
                out_k_n1_ho1_wo1_b_n2_ho2_wo2_global_merged_desc.GetOffsetFromMultiIndex(
                    k_thread_data_on_global, 0, 0, 0, b_thread_data_on_global, 0, 0, 0);

            threadwise_generic_tensor_slice_copy_v1(
                out_n0_n1_n2_k0_k1_k2_ho0_ho1_ho2_wo0_wo1_wo2_thread_desc,
                p_out_thread,
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                out_n0_n1_n2_k0_k1_k2_ho0_ho1_ho2_wo0_wo1_wo2_global_mem_desc,
                p_out_thread_on_global,
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                out_n0_n1_n2_k0_k1_k2_ho0_ho1_ho2_wo0_wo1_wo2_thread_desc.GetLengths(),
                arithmetic_sequence_gen<0, 12, 1>::type{},
                Number<1>{});
        }
    }
};

} // namespace ck
#endif
