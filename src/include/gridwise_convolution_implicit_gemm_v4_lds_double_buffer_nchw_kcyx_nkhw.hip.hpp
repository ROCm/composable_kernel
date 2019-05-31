#pragma once
#include "common.hip.hpp"
#include "ConstantTensorDescriptor.hip.hpp"
#include "ConstantMergedTensorDescriptor.hip.hpp"
#include "ConstantMatrixDescriptor.hip.hpp"
#include "blockwise_generic_tensor_slice_op.hip.hpp"
#include "blockwise_gemm.hip.hpp"
#include "threadwise_tensor_slice_op.hip.hpp"

// define B = merge(N, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          index_t BPerBlock,
          index_t KPerBlock,
          index_t EPerBlock,
          index_t N1,
          index_t N2,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class InBlockCopySubLengths_E_N1_B_N2,
          class InBlockCopyClusterLengths_E_N1_B_N2,
          index_t InBlockCopySrcDataPerRead_B,
          index_t InBlockCopyDstDataPerWrite_N2,
          class WeiBlockCopySubLengths_E_K,
          class WeiBlockCopyClusterLengths_E_K,
          index_t WeiBlockCopySrcDataPerRead_E,
          index_t WeiBlockCopyDstDataPerWrite_K>
struct GridwiseConvolutionImplicitGemm_v4_lds_double_buffer_nchw_kcyx_nkhw
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        // this is a mess
        // TODO: find more elegent way of specifying (or calculating) performance parameters
        static_assert(N2 == GemmNPerThreadSubC, "wrong!");
        static_assert((N1 * N2 * BPerBlock) %
                              (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};
        constexpr auto I7 = Number<7>{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto in_n_c_h_w_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_c_y_x_global_desc = WeiGlobalDesc{};
        constexpr auto out_n_k_h_w_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_c_h_w_global_desc.GetLength(I0);
        constexpr index_t C  = in_n_c_h_w_global_desc.GetLength(I1);
        constexpr index_t Hi = in_n_c_h_w_global_desc.GetLength(I2);
        constexpr index_t Wi = in_n_c_h_w_global_desc.GetLength(I3);

        constexpr index_t K  = out_n_k_h_w_global_desc.GetLength(I1);
        constexpr index_t Ho = out_n_k_h_w_global_desc.GetLength(I2);
        constexpr index_t Wo = out_n_k_h_w_global_desc.GetLength(I3);

        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

        static_assert(N % (N1 * N2) == 0, "wrong! cannot divice N evenly among thread");

        constexpr index_t N0 = N / (N1 * N2);

        constexpr index_t B = N0 * Ho * Wo;

        constexpr index_t E = C * Y * X;

        // divide block work by [K, B]
        static_assert(K % KPerBlock == 0 && B % BPerBlock == 0 && E % (2 * EPerBlock) == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t KBlockWork = K / KPerBlock;
        constexpr index_t BBlockWork = B / BPerBlock;

        constexpr auto block_work_desc =
            make_ConstantTensorDescriptor_default_rank_packed(Sequence<KBlockWork, BBlockWork>{});

        const auto block_work_multi_id =
            block_work_desc.GetMultiIndexFrom1dIndex(get_block_1d_id());

        const index_t k_block_data_on_global = block_work_multi_id[0] * KPerBlock;
        const index_t b_block_data_on_global = block_work_multi_id[1] * BPerBlock;

        // input tensor
        //     tensor descriptor in device memory [N0, N1, N2, Ho, Wo]
        constexpr auto in_n0_n1_n2_h_w_global_desc = in_n_c_h_w_global_desc.Slice(I2, Number<Ho>{})
                                                         .Slice(I3, Number<Wo>{})
                                                         .Fold(I0, Number<N1>{}, Number<N2>{})
                                                         .Extract(Sequence<0, 1, 2, 4, 5>{});

        //     batch descritpor for device memory
        constexpr auto in_c_y_x_global_desc = in_n_c_h_w_global_desc.Slice(I2, Number<Y>{})
                                                  .Slice(I3, Number<X>{})
                                                  .Extract(Sequence<1, 2, 3>{});

        //     merged tensor descriptor in device memory [E, N1, B, N2], src of blockwise copy
        constexpr auto in_e_n1_b_n2_global_merged_desc = make_ConstantMergedTensorDescriptor(
            in_c_y_x_global_desc.Embed(in_n0_n1_n2_h_w_global_desc),
            Sequence<0, 1, 2>{},
            Sequence<4>{},
            Sequence<3, 6, 7>{},
            Sequence<5>{});

#if 0
        if(get_block_1d_id() == 0 && get_thread_local_1d_id() == 0)
        {
            print_ConstantTensorDescriptor(in_n0_n1_n2_h_w_global_desc,
                                           "in_n0_n1_n2_h_w_global_desc: ");
            print_ConstantTensorDescriptor(in_c_y_x_global_desc, "in_c_y_x_global_desc: ");
            print_ConstantMergedTensorDescriptor(in_e_n1_b_n2_global_merged_desc,
                                                 "in_e_n1_b_n2_global_merged_desc: ");
        }
#endif

        //     memory layout descriptor in LDS [E, N1, B, N2], dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto in_e_n1_b_n2_block_desc = make_ConstantTensorDescriptor_default_rank_aligned(
            Sequence<EPerBlock, N1, BPerBlock, N2>{}, Number<InBlockCopyDstDataPerWrite_N2>{});

        //     this check is ad-hoc
        //     TODO: need to properly implement tensor descriptor with multiple alignment
        //     requirements
        static_assert(in_e_n1_b_n2_block_desc.GetStride(I1) % GemmDataPerReadB == 0,
                      "GemmDataPerReadB alignment requirement is not satisfied");

        // input blockwise copy
        //     slice a merged tensor, reorder and copy to a normal tensor
        //     this copy operator already has blockwise offset built-in
        auto blockwise_in_copy = BlockwiseGenericTensorSliceCopy_v1<
            BlockSize,
            Float,
            decltype(in_e_n1_b_n2_global_merged_desc),
            decltype(in_e_n1_b_n2_block_desc),
            decltype(in_e_n1_b_n2_block_desc.GetLengths()),
            InBlockCopySubLengths_E_N1_B_N2,
            InBlockCopyClusterLengths_E_N1_B_N2,
            Sequence<0, 1, 3, 2>, // thread_arrange_order [E, N1, N2, B]
            Sequence<0, 1, 3, 2>, // src_access_order [E, N1, N2, B]
            Sequence<0, 1, 2, 3>, // dst_access_order [E, N1, B, N2]
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_N2>({0, 0, b_block_data_on_global, 0}, {0, 0, 0, 0});

        // weight tensor
        //     tensor descriptor in device memory, src of blockwise copy
        constexpr auto wei_e_k_global_desc =
            wei_k_c_y_x_global_desc.Unfold(I1, I3).ReorderGivenNew2Old(Sequence<1, 0>{});

        //     tensor descriptor in LDS, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto wei_e_k_block_desc = make_ConstantTensorDescriptor_default_rank_aligned(
            Sequence<EPerBlock, KPerBlock>{},
            Number<mod_conv::max(WeiBlockCopyDstDataPerWrite_K, GemmDataPerReadA)>{});

        // operator for blockwise copy of weight into LDS
        //     slice a tensor, and copy it into another tensor
        //     this copy operator already have blockwise offset built-in
        auto blockwise_wei_copy =
            BlockwiseGenericTensorSliceCopy_v1<BlockSize,
                                               Float,
                                               decltype(wei_e_k_global_desc),
                                               decltype(wei_e_k_block_desc),
                                               decltype(wei_e_k_block_desc.GetLengths()),
                                               WeiBlockCopySubLengths_E_K,
                                               WeiBlockCopyClusterLengths_E_K,
                                               Sequence<1, 0>, // thread_arrange_order [K, E]
                                               Sequence<1, 0>, // src_access_order [K, E]
                                               Sequence<0, 1>, // dst_access_order [E, K]
                                               WeiBlockCopySrcDataPerRead_E,
                                               WeiBlockCopyDstDataPerWrite_K>(
                {0, k_block_data_on_global}, {0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[EPerBlock, KPerBlock] is in LDS
        //     b_mtx[EPerBlocl, N1 * BPerBlock * N2] is in LDS
        //     c_mtx[KPerBlock, N1 * BPerBlock * N2] is distributed among threads, and saved in
        //     register
        constexpr auto a_e_k_block_mtx_desc = make_ConstantMatrixDescriptor(
            Number<EPerBlock>{}, Number<KPerBlock>{}, Number<wei_e_k_block_desc.GetStride(I0)>{});

        constexpr auto b_e_n1bn2_block_mtx_desc =
            make_ConstantMatrixDescriptor(Number<EPerBlock>{},
                                          Number<N1 * BPerBlock * N2>{},
                                          Number<in_e_n1_b_n2_block_desc.GetStride(I0)>{});

        // sanity check
        static_assert(KPerBlock % (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr index_t GemmMRepeat =
            KPerBlock / (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_k0k2_n1n2_thread_mtx_desc = make_ConstantMatrixDescriptor(
            Number<GemmMRepeat * GemmMPerThreadSubC>{}, Number<N1 * N2>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
            BlockSize,
            decltype(a_e_k_block_mtx_desc),
            decltype(b_e_n1bn2_block_mtx_desc),
            decltype(c_k0k2_n1n2_thread_mtx_desc),
            GemmMPerThreadSubC,
            GemmNPerThreadSubC,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmKPerThreadLoop,
            GemmDataPerReadA,
            GemmDataPerReadB>{};

        // choose GEMM implementation here
        const auto run_blockwise_gemm = [&](auto... Xs) {
#if 1
            return blockwise_gemm.Run(Xs...);
#elif 0
            return blockwise_gemm.Run_asm(Xs...);
#endif
        };

        // LDS allocation for input and weight: be careful of alignment
        constexpr index_t max_align = mod_conv::max(InBlockCopyDstDataPerWrite_N2,
                                                    WeiBlockCopyDstDataPerWrite_K,
                                                    GemmDataPerReadA,
                                                    GemmDataPerReadB);

        constexpr index_t in_block_space =
            in_e_n1_b_n2_block_desc.GetElementSpace(Number<max_align>{});

        constexpr index_t wei_block_space = wei_e_k_block_desc.GetElementSpace(Number<max_align>{});

        __shared__ Float p_in_block_double[2 * in_block_space];
        __shared__ Float p_wei_block_double[2 * wei_block_space];

        // register allocation for output
        Float p_out_thread[c_k0k2_n1n2_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_k0k2_n1n2_thread_mtx_desc, p_out_thread);

#if 0
        if(e == 0 * EPerBlock && get_block_1d_id() == 0)
        {
            printf("id %5u %5u: "
                   "mThreadSrcOffset %u, mThreadDstOffset %u \n",
                   get_block_1d_id(),
                   get_thread_local_1d_id(),
                   blockwise_wei_copy.mThreadSrcOffset,
                   blockwise_wei_copy.mThreadDstOffset);
        }
#endif

        // LDS double buffer: preload data into LDS
        {
            Float p_in_register_clipboard[blockwise_in_copy.GetRegisterClipboardSize()];
            Float p_wei_register_clipboard[blockwise_wei_copy.GetRegisterClipboardSize()];

            blockwise_in_copy.RunLoadRegisterClipboard(p_in_global, p_in_register_clipboard);
            blockwise_wei_copy.RunLoadRegisterClipboard(p_wei_global, p_wei_register_clipboard);

            blockwise_in_copy.RunStoreRegisterClipboard(p_in_register_clipboard, p_in_block_double);
            blockwise_wei_copy.RunStoreRegisterClipboard(p_wei_register_clipboard,
                                                         p_wei_block_double);
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

                Float p_in_register_clipboard[blockwise_in_copy.GetRegisterClipboardSize()];
                Float p_wei_register_clipboard[blockwise_wei_copy.GetRegisterClipboardSize()];

                blockwise_in_copy.MoveSlicingWindowOnSourceTensor(I0, Number<EPerBlock>{}, True);
                blockwise_wei_copy.MoveSlicingWindowOnSourceTensor(I0, Number<EPerBlock>{}, True);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                blockwise_in_copy.RunLoadRegisterClipboard(p_in_global, p_in_register_clipboard);
                blockwise_wei_copy.RunLoadRegisterClipboard(p_wei_global, p_wei_register_clipboard);

                // LDS double buffer: GEMM on current data
                run_blockwise_gemm(p_wei_block_now, p_in_block_now, p_out_thread);

                // LDS double buffer: store next data to LDS
                blockwise_in_copy.RunStoreRegisterClipboard(p_in_register_clipboard,
                                                            p_in_block_next);
                blockwise_wei_copy.RunStoreRegisterClipboard(p_wei_register_clipboard,
                                                             p_wei_block_next);
            }
        }

        // LDS double buffer: tail
        {
            Float p_in_register_clipboard[blockwise_in_copy.GetRegisterClipboardSize()];
            Float p_wei_register_clipboard[blockwise_wei_copy.GetRegisterClipboardSize()];

            // even iteration
            blockwise_in_copy.MoveSlicingWindowOnSourceTensor(I0, Number<EPerBlock>{}, True);
            blockwise_wei_copy.MoveSlicingWindowOnSourceTensor(I0, Number<EPerBlock>{}, True);

            __syncthreads();

            // LDS doubel buffer: load next data from device mem
            blockwise_in_copy.RunLoadRegisterClipboard(p_in_global, p_in_register_clipboard);
            blockwise_wei_copy.RunLoadRegisterClipboard(p_wei_global, p_wei_register_clipboard);

            // LDS double buffer: GEMM on current data
            run_blockwise_gemm(p_wei_block_double, p_in_block_double, p_out_thread);

            // LDS double buffer: store next data to LDS
            blockwise_in_copy.RunStoreRegisterClipboard(p_in_register_clipboard,
                                                        p_in_block_double + in_block_space);
            blockwise_wei_copy.RunStoreRegisterClipboard(p_wei_register_clipboard,
                                                         p_wei_block_double + wei_block_space);

            // odd iteration
            __syncthreads();

            // LDS double buffer: GEMM on current data
            run_blockwise_gemm(p_wei_block_double + wei_block_space,
                               p_in_block_double + in_block_space,
                               p_out_thread);
        }

        // copy output: register to global memory
        {
            constexpr index_t K2 = GemmMPerThreadSubC;
            constexpr index_t K1 = GemmMLevel0Cluster * GemmMLevel1Cluster;
            constexpr index_t K0 = K / (K1 * K2);

            // define tensor descriptor for threadwise copy
            //     output memory layout descriptor in register
            constexpr auto out_k0_k1_k2_n1_n0_h_w_n2_thread_mem_desc =
                make_ConstantTensorDescriptor_default_rank_packed(
                    Sequence<KPerBlock / (K1 * K2), 1, K2, N1, 1, 1, 1, N2>{});

            //     output tensor descriptor in register, src of threadwise copy
            constexpr auto out_n0_n1_n2_k0_k1_k2_h_w_thread_desc =
                out_k0_k1_k2_n1_n0_h_w_n2_thread_mem_desc.ReorderGivenNew2Old(
                    Sequence<4, 3, 7, 0, 1, 2, 5, 6>{});

            //     output memory layout descriptor in device memory, dst of threadwise copy
            constexpr auto out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc =
                out_n_k_h_w_global_desc.Fold(I1, Number<K1>{}, Number<K2>{})
                    .Fold(I0, Number<N1>{}, Number<N2>{});

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t k_thread_data_on_global =
                k_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t b_thread_data_on_global =
                b_block_data_on_global + c_thread_mtx_on_block.col / N2;

            //     output merged global tensor descriptor, for calculating origin of thread tensor
            //     in global memory
            constexpr auto out_k_n1_b_n2_global_merged_desc = make_ConstantMergedTensorDescriptor(
                out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc.Unfold(I3, I5),
                Sequence<3>{},
                Sequence<1>{},
                Sequence<0, 4, 5>{},
                Sequence<2>{});

            //     origin of dst in device memory
            Float* p_out_thread_on_global =
                p_out_global +
                out_k_n1_b_n2_global_merged_desc.GetOffsetFromMultiIndex(
                    k_thread_data_on_global, 0, b_thread_data_on_global, 0);

            threadwise_generic_tensor_slice_copy(out_n0_n1_n2_k0_k1_k2_h_w_thread_desc,
                                                 p_out_thread,
                                                 {0, 0, 0, 0, 0, 0, 0, 0},
                                                 out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc,
                                                 p_out_thread_on_global,
                                                 {0, 0, 0, 0, 0, 0, 0, 0},
                                                 out_n0_n1_n2_k0_k1_k2_h_w_thread_desc.GetLengths(),
                                                 arithmetic_sequence_gen<0, 8, 1>::SeqType{});
        }
    }
};
