#pragma once
#include "common.hip.hpp"
#include "ConstantTensorDescriptor.hip.hpp"
#include "ConstantMergedTensorDescriptor.hip.hpp"
#include "ConstantMatrixDescriptor.hip.hpp"
#include "blockwise_merged_tensor_slice_op.hip.hpp"
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
          index_t CPerBlock,
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
          class InBlockCopySubLengths_N1_N2_C_B,
          class InBlockCopyClusterLengths_N1_N2_C_B,
          index_t InBlockCopySrcDataPerRead_B,
          index_t InBlockCopyDstDataPerWrite_N2,
          class WeiBlockCopySubLengths_C_K,
          class WeiBlockCopyClusterLengths_C_K,
          index_t WeiBlockCopyDataPerAccess_K>
struct GridwiseConvolutionImplicitGemm_v3_nchw_cyxk_nkhw
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        // this is a mess
        // TODO: fidn more elegent way of specifying (or calculating) performance parameters
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

        constexpr auto in_n_c_h_w_global_desc  = InGlobalDesc{};
        constexpr auto wei_c_y_x_k_global_desc = WeiGlobalDesc{};
        constexpr auto out_n_k_h_w_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_c_h_w_global_desc.GetLength(I0);
        constexpr index_t C  = in_n_c_h_w_global_desc.GetLength(I1);
        constexpr index_t Hi = in_n_c_h_w_global_desc.GetLength(I2);
        constexpr index_t Wi = in_n_c_h_w_global_desc.GetLength(I3);

        constexpr index_t K  = out_n_k_h_w_global_desc.GetLength(I1);
        constexpr index_t Ho = out_n_k_h_w_global_desc.GetLength(I2);
        constexpr index_t Wo = out_n_k_h_w_global_desc.GetLength(I3);

        constexpr index_t Y = wei_c_y_x_k_global_desc.GetLength(I1);
        constexpr index_t X = wei_c_y_x_k_global_desc.GetLength(I2);

        static_assert(N % (N1 * N2) == 0, "wrong! cannot divice N evenly among thread");

        constexpr index_t N0 = N / (N1 * N2);

        constexpr index_t B = N0 * Ho * Wo;

        // divide block work by [K, B]
        static_assert(K % KPerBlock == 0 && B % BPerBlock == 0 && C % CPerBlock == 0,
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
        //     memory layout descriptor in device memory [N0, N1, N2, C, H, W]
        constexpr auto in_n0_n1_n2_c_h_w_global_mem_desc =
            in_n_c_h_w_global_desc.Fold(I0, Number<N1>{}, Number<N2>{});

        //     merged tensor descriptor in device memory [N1, N2, C, B], src of blockwise copy
        constexpr auto in_n1_n2_c_b_global_merged_desc = make_ConstantMergedTensorDescriptor(
            in_n0_n1_n2_c_h_w_global_mem_desc.ReorderGivenNew2Old(Sequence<1, 2, 3, 0, 4, 5>{})
                .Slice(I4, Number<Ho>{})
                .Slice(I5, Number<Wo>{}),
            Sequence<0>{},
            Sequence<1>{},
            Sequence<2>{},
            Sequence<3, 4, 5>{});

        //     memory layout descriptor in LDS [C, N1, B, N2]
        //     be careful of LDS alignment
        constexpr auto in_c_n1_b_n2_block_mem_desc =
            make_ConstantTensorDescriptor_default_rank_aligned(
                Sequence<CPerBlock, N1, BPerBlock, N2>{}, Number<InBlockCopyDstDataPerWrite_N2>{});

        //    tensor descriptor in LDS [N1, N2, C, B], dst of blockwise copy
        constexpr auto in_n1_n2_c_b_block_desc =
            in_c_n1_b_n2_block_mem_desc.ReorderGivenNew2Old(Sequence<1, 3, 0, 2>{});

        //     this check is ad-hoc
        //     TODO: need to properly implement tensor descriptor with alignment
        static_assert(in_c_n1_b_n2_block_mem_desc.GetStride(I1) % GemmDataPerReadB == 0,
                      "GemmDataPerReadB alignment requirement is not satisfied");

        // input blockwise copy
        //     slice a merged tensor, reorder and copy to a normal tensor
        //     this copy operator already has blockwise offset built-in
        const auto blockwise_in_copy = BlockwiseTensorSliceCopy_generic_v1<
            BlockSize,
            Float,
            decltype(in_n1_n2_c_b_global_merged_desc),
            decltype(in_n1_n2_c_b_block_desc),
            decltype(in_n1_n2_c_b_block_desc.GetLengths()),
            InBlockCopySubLengths_N1_N2_C_B,
            InBlockCopyClusterLengths_N1_N2_C_B,
            Sequence<2, 0, 1, 3>, // thread_arrange_order [C, N1, N2, B]
            Sequence<0, 1, 2, 3>, // src_access_order [N1, N2, C, B]
            Sequence<2, 0, 3, 1>, // dst_access_order [C, N1, B, N2]
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_N2>({0, 0, 0, b_block_data_on_global}, {0, 0, 0, 0});

        // weight tensor
        //     tensor descriptor in device memory, src of blockwise copy
        constexpr auto wei_c_k_global_desc = wei_c_y_x_k_global_desc.Extract(I0, I3);

        //     tensor descriptor in LDS, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto wei_c_k_block_desc = make_ConstantTensorDescriptor_default_rank_aligned(
            Sequence<CPerBlock, KPerBlock>{},
            Number<mod_conv::max(WeiBlockCopyDataPerAccess_K, GemmDataPerReadA)>{});

        // operator for blockwise copy of weight into LDS
        //     slicing a tensor
        //     this copy operator already have blockwise offset built-in
        const auto blockwise_wei_copy =
            BlockwiseTensorSliceCopy_generic_v1<BlockSize,
                                                Float,
                                                decltype(wei_c_k_global_desc),
                                                decltype(wei_c_k_block_desc),
                                                decltype(wei_c_k_block_desc.GetLengths()),
                                                WeiBlockCopySubLengths_C_K,
                                                WeiBlockCopyClusterLengths_C_K,
                                                Sequence<0, 1>, // thread_arrange_order [C, K]
                                                Sequence<0, 1>, // src_access_order [C, K]
                                                Sequence<0, 1>, // dst_access_order [C, K]
                                                WeiBlockCopyDataPerAccess_K,
                                                WeiBlockCopyDataPerAccess_K>(
                {0, k_block_data_on_global}, {0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[CPerBlock, KPerBlock] is in LDS
        //     b_mtx[CPerBlocl, N1 * BPerBlock * N2] is in LDS
        //     c_mtx[KPerBlock, N1 * BPerBlock * N2] is distributed among threads, and saved in
        //     register
        constexpr auto a_c_k_block_mtx_desc = make_ConstantMatrixDescriptor(
            Number<CPerBlock>{}, Number<KPerBlock>{}, Number<wei_c_k_block_desc.GetStride(I0)>{});

        constexpr auto b_c_n1bn2_block_mtx_desc =
            make_ConstantMatrixDescriptor(Number<CPerBlock>{},
                                          Number<N1 * BPerBlock * N2>{},
                                          Number<in_c_n1_b_n2_block_mem_desc.GetStride(I0)>{});

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
            decltype(a_c_k_block_mtx_desc),
            decltype(b_c_n1bn2_block_mtx_desc),
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

        // LDS allocation for input and weight: be careful of alignment
        constexpr index_t max_align = mod_conv::max(InBlockCopyDstDataPerWrite_N2,
                                                    WeiBlockCopyDataPerAccess_K,
                                                    GemmDataPerReadA,
                                                    GemmDataPerReadB);

        constexpr index_t in_block_space =
            in_c_n1_b_n2_block_mem_desc.GetElementSpace(Number<max_align>{});

        constexpr index_t wei_block_space = wei_c_k_block_desc.GetElementSpace(Number<max_align>{});

        __shared__ Float p_in_block[in_block_space];
        __shared__ Float p_wei_block[wei_block_space];

        // register allocation for output
        Float p_out_thread[c_k0k2_n1n2_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_k0k2_n1n2_thread_mtx_desc, p_out_thread);

        // do work
        for(index_t y = 0; y < Y; ++y)
        {
            for(index_t x = 0; x < X; ++x)
            {
                // calculate origin of block input and weight tensor on global memory
                const Float* p_in_block_on_global =
                    p_in_global + in_n_c_h_w_global_desc.GetOffsetFromMultiIndex(0, 0, y, x);

                const Float* p_wei_block_on_global =
                    p_wei_global + wei_c_y_x_k_global_desc.GetOffsetFromMultiIndex(0, y, x, 0);

                for(index_t
                        c_block_data_on_global = 0;
                    c_block_data_on_global < C;
                    c_block_data_on_global += CPerBlock,
                        p_in_block_on_global += CPerBlock * in_n_c_h_w_global_desc.GetStride(I1),
                        p_wei_block_on_global += CPerBlock * wei_c_y_x_k_global_desc.GetStride(I0))
                {
#if 1 // debug
                    blockwise_in_copy.Run(p_in_block_on_global, p_in_block);
                    blockwise_wei_copy.Run(p_wei_block_on_global, p_wei_block);
#endif

                    __syncthreads();

#if 1 // debug
                    blockwise_gemm.Run(p_wei_block, p_in_block, p_out_thread);
#endif

                    __syncthreads();
                }
            }
        }

        // copy output: register to global memory
        {
            constexpr index_t K2 = GemmMPerThreadSubC;
            constexpr index_t K1 = GemmMLevel0Cluster * GemmMLevel1Cluster;
            constexpr index_t K0 = K / (K1 * K2);

            // define tensor descriptor for threadwise copy
            //     output tensor (also, memory layout) descriptor in register, src of threadwise
            //     copy
            constexpr auto out_k0_k1_k2_n1_b_n2_thread_mem_desc =
                make_ConstantTensorDescriptor_default_rank_packed(
                    Sequence<KPerBlock / (K1 * K2), 1, K2, N1, 1, N2>{});

            //     output memory layout descriptor in device memory
            constexpr auto out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc =
                out_n_k_h_w_global_desc.Fold(I1, Number<K1>{}, Number<K2>{})
                    .Fold(I0, Number<N1>{}, Number<N2>{});

            //     output merged tensor descriptor in device memory, dst of threadwise copy
            constexpr auto out_k0_k1_k2_n1_b_n2_global_merged_desc =
                make_ConstantMergedTensorDescriptor(
                    out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc.ReorderGivenNew2Old(
                        Sequence<3, 4, 5, 1, 0, 6, 7, 2>{}),
                    Sequence<0>{},
                    Sequence<1>{},
                    Sequence<2>{},
                    Sequence<3>{},
                    Sequence<4, 5, 6>{},
                    Sequence<7>{});

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            //     origin of thread tensor on global
            const index_t k_thread_data_on_global =
                k_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t b_thread_data_on_global =
                b_block_data_on_global + c_thread_mtx_on_block.col / N2;

//     output merged global tensor descriptor, for calculating origin of thread tensor
//     in global memory
#if 0 // unfold a merged tensor is not implemented yet
            constexpr auto out_k_n1_b_n2_global_merged_desc =
                out_k0_k1_k2_n1_b_n2_global_merged_desc.Unfold(I0, I2);
#else
            constexpr auto out_k_n1_b_n2_global_merged_desc = make_ConstantMergedTensorDescriptor(
                out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc
                    .ReorderGivenNew2Old(Sequence<3, 4, 5, 1, 0, 6, 7, 2>{})
                    .Unfold(I0, I2),
                Sequence<0>{},
                Sequence<1>{},
                Sequence<2, 3, 4>{},
                Sequence<5>{});
#endif

            //     origin of thread tensor in global memory
            Float* p_out_thread_on_global =
                p_out_global +
                out_k_n1_b_n2_global_merged_desc.GetOffsetFromMultiIndex(
                    k_thread_data_on_global, 0, 0, 0); // dst origin on merged global tensor

            threadwise_tensor_slice_copy_generic(
                out_k0_k1_k2_n1_b_n2_thread_mem_desc, // src thread tensor (in register) descriptor
                p_out_thread,                         // origin of src
                {0, 0, 0, 0, 0, 0}, // starting point of slice, w.r.t. origin of src
                out_k0_k1_k2_n1_b_n2_global_merged_desc, // dst global merged tensor (in device mem)
                                                         // descriptor
                p_out_thread_on_global,                  // origin of dst
                {0,
                 0,
                 0,
                 0,
                 b_thread_data_on_global,
                 0}, // starting point of slice w.r.t. origin of dst
                out_k0_k1_k2_n1_b_n2_thread_mem_desc.GetLengths(), // slice lengths
                Sequence<3, 5, 0, 1, 2, 4>{} // dimension access order [n1, n2, k0, k1, k2, b]
                );

#if 0
            if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
            {
                print_ConstantTensorDescriptor(in_n0_n1_n2_c_h_w_global_mem_desc,
                                               "in_n0_n1_n2_c_h_w_global_mem_desc");

                print_ConstantMergedTensorDescriptor(in_n1_n2_c_b_global_merged_desc,
                                                     "in_n1_n2_c_b_global_merged_desc");

                print_ConstantTensorDescriptor(in_c_n1_b_n2_block_mem_desc,
                                               "in_c_n1_b_n2_block_mem_desc");

                print_ConstantTensorDescriptor(in_n1_n2_c_b_block_desc, "in_n1_n2_c_b_block_desc");

                print_ConstantTensorDescriptor(out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc,
                                               "out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc");

                print_ConstantMergedTensorDescriptor(out_k_n1_b_n2_global_merged_desc,
                                                     "out_k_n1_b_n2_global_merged_desc");

                print_ConstantTensorDescriptor(out_k0_k1_k2_n1_b_n2_thread_mem_desc,
                                               "out_k0_k1_k2_n1_b_n2_thread_mem_desc");
            }
#endif
        }
    }
};
