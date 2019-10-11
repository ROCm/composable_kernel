#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V1R3_NCHW_CYXK_NKHW
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V1R3_NCHW_CYXK_NKHW

#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_2d_tensor_op.hpp"
#include "blockwise_tensor_slice_copy.hpp"
#include "threadwise_tensor_slice_copy.hpp"
#include "threadwise_generic_tensor_op.hpp"
#include "blockwise_batched_gemm.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t CPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t NPerThread,
          index_t KPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class InBlockReorderSrcSubLengths_NCHW,
          class InBlockReorderSrcClusterLengths_NCHW,
          class InBlockReorderMapThreadCluster2SrcCluster_CHNW2NCHW,
          index_t InBlockReorderDataPerRead_W,
          index_t InBlockReorderDataPerWrite_N,
          class WeiBlockCopyClusterLengths_CK, // not used
          index_t WeiBlockCopyDataPerRead_K,
          index_t OutThreadCopyDataPerWrite_W>
struct GridwiseConvolutionImplicitGemm_v1r3_nchw_cyxk_nkhw
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        // be careful of this assertion
        static_assert(
            NPerBlock % NPerThread == 0 &&
                ((GemmNPerThreadSubC <= NPerBlock && NPerBlock % GemmNPerThreadSubC == 0) ||
                 (GemmNPerThreadSubC >= NPerBlock && NPerThread == NPerBlock &&
                  GemmNPerThreadSubC % NPerThread == 0)),
            "wrong!");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto in_n_c_h_w_global_desc  = InGlobalDesc{};
        constexpr auto wei_c_y_x_k_global_desc = WeiGlobalDesc{};
        constexpr auto out_n_k_h_w_global_desc = OutGlobalDesc{};

        constexpr index_t C = in_n_c_h_w_global_desc.GetLength(I1);

        constexpr index_t N  = out_n_k_h_w_global_desc.GetLength(I0);
        constexpr index_t K  = out_n_k_h_w_global_desc.GetLength(I1);
        constexpr index_t Ho = out_n_k_h_w_global_desc.GetLength(I2);
        constexpr index_t Wo = out_n_k_h_w_global_desc.GetLength(I3);

        constexpr index_t Y = wei_c_y_x_k_global_desc.GetLength(I1);
        constexpr index_t X = wei_c_y_x_k_global_desc.GetLength(I2);

        // divide block work: [N, K, Ho, Wo]
        static_assert(N % NPerBlock == 0 && K % KPerBlock == 0 && C % CPerBlock == 0 &&
                          Ho % HoPerBlock == 0 && Wo % WoPerBlock == 0,
                      "wrong! cannot evenly divide work for workgroup ");

        constexpr index_t NBlockWork = math::integer_divide_ceil(N, NPerBlock);
        constexpr index_t KBlockWork = math::integer_divide_ceil(K, KPerBlock);
        constexpr index_t HBlockWork = math::integer_divide_ceil(Ho, HoPerBlock);
        constexpr index_t WBlockWork = math::integer_divide_ceil(Wo, WoPerBlock);

        constexpr auto block_work_desc = make_ConstantTensorDescriptor_packed(
            Sequence<NBlockWork, KBlockWork, HBlockWork, WBlockWork>{});

        const auto block_work_multi_id =
            block_work_desc.GetMultiIndexFrom1dIndex(get_block_1d_id());

        const index_t n_block_data_begin  = block_work_multi_id[0] * NPerBlock;
        const index_t k_block_data_begin  = block_work_multi_id[1] * KPerBlock;
        const index_t ho_block_data_begin = block_work_multi_id[2] * HoPerBlock;
        const index_t wo_block_data_begin = block_work_multi_id[3] * WoPerBlock;

        const index_t hi_block_data_begin = ho_block_data_begin;
        const index_t wi_block_data_begin = wo_block_data_begin;

        // global tensor view
        constexpr auto wei_c_k_global_desc =
            make_ConstantTensorDescriptor(Sequence<C, K>{}, Sequence<Y * X * K, 1>{});

        // LDS tensor view
        //   be careful of alignment
        constexpr index_t max_align = math::lcm(InBlockReorderDataPerWrite_N,
                                                WeiBlockCopyDataPerRead_K,
                                                GemmDataPerReadA,
                                                GemmDataPerReadB);

        constexpr auto in_c_h_w_n_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, HoPerBlock, WoPerBlock, NPerBlock>{},
            Number<InBlockReorderDataPerWrite_N>{});

        // this check is ad-hoc
        // TODO: need to properly implement tensor descriptor with alignment
        static_assert(in_c_h_w_n_block_desc.GetStride(I1) % GemmDataPerReadB == 0,
                      "GemmDataPerReadB alignment requirement is not meet");

        constexpr auto wei_c_k_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, KPerBlock>{},
            Number<math::lcm(WeiBlockCopyDataPerRead_K, GemmDataPerReadA)>{});

        // tensor view of threadwise output in register
        constexpr auto out_k_h_w_n_thread_desc = make_ConstantTensorDescriptor_packed(
            Sequence<KPerThread, HoPerThread, WoPerThread, NPerThread>{});

        // blockwise copy
        // input: format is [N, C, Hi, Wi] to [C, Hi, Wi, N]
        constexpr auto map_chwn2nchw = Sequence<1, 2, 3, 0>{};

        const auto blockwise_in_copy_reorder = BlockwiseTensorSliceReorderCopy_v3<
            BlockSize,
            Float,
            decltype(in_n_c_h_w_global_desc),
            decltype(in_c_h_w_n_block_desc),
            Sequence<NPerBlock, CPerBlock, HoPerBlock, WoPerBlock>,
            InBlockReorderSrcSubLengths_NCHW,
            InBlockReorderSrcClusterLengths_NCHW,
            decltype(map_chwn2nchw),
            InBlockReorderMapThreadCluster2SrcCluster_CHNW2NCHW,
            InBlockReorderDataPerRead_W,
            InBlockReorderDataPerWrite_N>({0, 0, 0, 0}, {0, 0, 0, 0});

        // blockwise wei copy
        //   format is [CPerBlock, KPerBlock]
        const auto blockwise_wei_copy =
            Blockwise2dTensorCopy3<BlockSize,
                                   Float,
                                   decltype(wei_c_k_global_desc),
                                   decltype(wei_c_k_block_desc),
                                   decltype(wei_c_k_block_desc.GetLengths()),
                                   WeiBlockCopyDataPerRead_K>({0, 0}, {0, 0});

        // a series of blockwise batched GEMM
        // C_matrix += transpose(A_matrix) * B_matrix
        //   A_matrix and B_matrix saved in LDS, C_matrix saved in register
        //   A_matrix[C,K] is a sub-matrix of wei_block[C,K]
        //   B_matrix[C,Wo*N] is a sub-matrix of in_block[C,Hi,Wi,N]
        //   C_matrix[K,Wo*N] is a sub-matrix of out_block[K,Ho,Wo,N]
        constexpr auto a_c_k_block_mtx_desc = make_ConstantMatrixDescriptor(
            Number<CPerBlock>{}, Number<KPerBlock>{}, Number<wei_c_k_block_desc.GetStride(I0)>{});

        constexpr auto b_c_wn_block_mtx_desc =
            make_ConstantMatrixDescriptor(Number<CPerBlock>{},
                                          Number<WoPerBlock * NPerBlock>{},
                                          Number<in_c_h_w_n_block_desc.GetStride(I0)>{});

        constexpr auto c_k_wn_thread_mtx_desc =
            make_ConstantMatrixDescriptor(Number<KPerThread>{},
                                          Number<WoPerThread * NPerThread>{},
                                          Number<out_k_h_w_n_thread_desc.GetStride(I0)>{});

        const auto blockwise_batch_gemm =
            BlockwiseBatchGemmBlockABlockBThreadCTransANormalBNormalC_V2<
                BlockSize,
                decltype(a_c_k_block_mtx_desc),
                decltype(b_c_wn_block_mtx_desc),
                decltype(c_k_wn_thread_mtx_desc),
                0,
                in_c_h_w_n_block_desc.GetStride(I1),
                out_k_h_w_n_thread_desc.GetStride(I1),
                HoPerBlock,
                GemmMPerThreadSubC,
                GemmNPerThreadSubC,
                GemmMLevel0Cluster,
                GemmNLevel0Cluster,
                GemmMLevel1Cluster,
                GemmNLevel1Cluster,
                GemmKPerThreadLoop,
                HoPerThread,
                GemmDataPerReadA,
                GemmDataPerReadB>{};

        // choose GEMM implementation here
        const auto run_blockwise_batch_gemm = [&](auto... Xs) {
#if 1
            return blockwise_batch_gemm.Run(Xs...);
#elif 0
            return blockwise_batch_gemm.Run_amd_asm(Xs...);
#else
            return blockwise_batch_gemm.Run_asm_v2(Xs...);
#endif
        };

        // LDS: be careful of alignment
        constexpr index_t in_block_space =
            in_c_h_w_n_block_desc.GetElementSpace(Number<max_align>{});
        constexpr index_t wei_block_space = wei_c_k_block_desc.GetElementSpace(Number<max_align>{});

        __shared__ Float p_in_block[in_block_space];
        __shared__ Float p_wei_block[wei_block_space];

        // register
        // C++ lambda doesn't capture array, use pointer instead
        Float p_out_thread_data[out_k_h_w_n_thread_desc.GetElementSpace()];
        Float* const p_out_thread = p_out_thread_data;

#if 0
        if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
        {
            print_ConstantTensorDescriptor(in_c_h_w_n_global_desc, "in_c_h_w_n_global_desc");
            print_ConstantTensorDescriptor(wei_c_y_x_k_global_desc, "wei_c_y_x_k_global_desc");

            print_ConstantTensorDescriptor(in_c_h_w_n_block_desc, "in_c_h_w_n_block_desc");
            print_ConstantTensorDescriptor(wei_c_k_block_desc, "wei_c_k_block_desc");

            printf("in_block_space %u, wei_block_space %u\n", in_block_space, wei_block_space);
        }
#endif

        // set threadwise output tensor to 0
        threadwise_generic_tensor_set_zero(out_k_h_w_n_thread_desc, p_out_thread);

#if 0
        const Float* p_in_global_block_offset =
            p_in_global +
            in_n_c_h_w_global_desc.GetOffsetFromMultiIndex(
                n_block_data_begin, 0, hi_block_data_begin, wi_block_data_begin);

        const Float* p_wei_global_block_offset =
            p_wei_global + wei_c_y_x_k_global_desc.GetOffsetFromMultiIndex(0, 0, 0, k_block_data_begin);

        for(index_t c_block_data_begin = 0; c_block_data_begin < C; c_block_data_begin += CPerBlock,
                    p_in_global_block_offset += CPerBlock * in_n_c_h_w_global_desc.GetStride(I1),
                    p_wei_global_block_offset += CPerBlock * wei_c_y_x_k_global_desc.GetStride(I0))
        {
            for(index_t y = 0; y < Y; ++y)
            {
                for(index_t x = 0; x < X; ++x)
                {
                    blockwise_in_copy_reorder.Run(p_in_global_block_offset +
                                                      in_n_c_h_w_global_desc.GetOffsetFromMultiIndex(0, 0, y, x),
                                                  p_in_block);

                    blockwise_wei_copy.Run(p_wei_global_block_offset +
                                               wei_c_y_x_k_global_desc.GetOffsetFromMultiIndex(0, y, x, 0),
                                           p_wei_block);

                    __syncthreads();

                    run_blockwise_batch_gemm(p_wei_block, p_in_block, p_out_thread);

                    __syncthreads();
                }
            }
        }
#else
        for(index_t y = 0; y < Y; ++y)
        {
            for(index_t x = 0; x < X; ++x)
            {
                const Float* p_in_global_block_offset =
                    p_in_global +
                    in_n_c_h_w_global_desc.GetOffsetFromMultiIndex(
                        n_block_data_begin, 0, hi_block_data_begin + y, wi_block_data_begin + x);

                const Float* p_wei_global_block_offset =
                    p_wei_global +
                    wei_c_y_x_k_global_desc.GetOffsetFromMultiIndex(0, y, x, k_block_data_begin);

                for(index_t c_block_data_begin = 0; c_block_data_begin < C;
                    c_block_data_begin += CPerBlock,
                            p_in_global_block_offset +=
                            CPerBlock * in_n_c_h_w_global_desc.GetStride(I1),
                            p_wei_global_block_offset +=
                            CPerBlock * wei_c_y_x_k_global_desc.GetStride(I0))
                {
                    blockwise_in_copy_reorder.Run(p_in_global_block_offset, p_in_block);

                    blockwise_wei_copy.Run(p_wei_global_block_offset, p_wei_block);

                    __syncthreads();

                    run_blockwise_batch_gemm(p_wei_block, p_in_block, p_out_thread);

                    __syncthreads();
                }
            }
        }
#endif

        // output: register to global mem,
        const auto c_thread_mtx_begin =
            blockwise_batch_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        const index_t k_thread_data_begin  = c_thread_mtx_begin.row;
        const index_t ho_thread_data_begin = c_thread_mtx_begin.batch;
        const index_t wo_thread_data_begin = c_thread_mtx_begin.col / NPerBlock;
        const index_t n_thread_data_begin  = c_thread_mtx_begin.col % NPerBlock;

        static_if<GemmNPerThreadSubC <= NPerBlock>{}([&](auto fwd) {
            // fwd do nothing but perfect forwarding.
            // Using this trick to make this lambda a generic lambda, so it won't be compiled until
            // begin instantiated here
            static_assert(
                (fwd(GemmNPerThreadSubC) <= NPerBlock && NPerBlock % GemmNPerThreadSubC == 0),
                "wrong!");

            // output is a 10d tensor
            constexpr index_t N2 = GemmNPerThreadSubC;
            constexpr index_t N1 = NPerBlock / N2;

            constexpr index_t W2 =
                (GemmNLevel0Cluster * GemmNLevel1Cluster) / fwd(NPerBlock / GemmNPerThreadSubC);
            constexpr index_t W1 = WoPerBlock / W2;

            constexpr index_t K2 = GemmMPerThreadSubC;
            constexpr index_t K1 = KPerBlock / KPerThread;

            constexpr auto out_10d_global_desc = fwd(out_n_k_h_w_global_desc)
                                                     .Fold(I3, Number<W1>{}, Number<W2>{})
                                                     .Fold(I1, Number<K1>{}, Number<K2>{})
                                                     .Fold(I0, Number<N1>{}, Number<N2>{});

            constexpr auto out_10d_thread_desc = fwd(out_k_h_w_n_thread_desc)
                                                     .Fold(I3, Number<1>{}, Number<N2>{})
                                                     .Fold(I2, Number<W1>{}, Number<1>{})
                                                     .Fold(I0, Number<1>{}, Number<K2>{});

#if 0
            if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
            {
                print_ConstantTensorDescriptor(out_k_h_w_n_thread_desc,
                                               "a: out_k_h_w_n_thread_desc");
                print_ConstantTensorDescriptor(out_10d_thread_desc, "a: out_10d_thread_desc");

                print_ConstantTensorDescriptor(out_n_k_h_w_global_desc,
                                               "a: out_n_k_h_w_global_desc");
                print_ConstantTensorDescriptor(out_10d_global_desc, "a: out_10d_global_desc");
            }
#endif

            constexpr auto map_out_global2thread = Sequence<7, 8, 9, 0, 1, 2, 3, 4, 5, 6>{};

            threadwise_tensor_slice_copy_reorder_given_dst2src_v2(
                out_10d_thread_desc,
                p_out_thread,
                out_10d_global_desc,
                p_out_global +
                    out_n_k_h_w_global_desc.GetOffsetFromMultiIndex(
                        n_block_data_begin + n_thread_data_begin,
                        k_block_data_begin + k_thread_data_begin,
                        ho_block_data_begin + ho_thread_data_begin,
                        wo_block_data_begin + wo_thread_data_begin),
                out_10d_thread_desc.GetLengths(),
                map_out_global2thread);
            // Number<OutThreadCopyDataPerWrite_W>{});
        }).Else([&](auto fwd) {
            static_assert(fwd(GemmNPerThreadSubC) >= NPerBlock && NPerThread == NPerBlock &&
                              GemmNPerThreadSubC % NPerThread == 0,
                          "wrong!");

            // output is a 10d tensor
            constexpr index_t N1 = NPerBlock;

            constexpr index_t W3 = GemmNPerThreadSubC / NPerBlock;
            constexpr index_t W2 = GemmNLevel0Cluster * GemmNLevel1Cluster;
            constexpr index_t W1 = WoPerBlock / fwd(W2 * W3);

            constexpr index_t K2 = GemmMPerThreadSubC;
            constexpr index_t K1 = KPerBlock / KPerThread;

            constexpr auto out_10d_global_desc =
                fwd(out_n_k_h_w_global_desc)
                    .Fold(I3, Number<W1>{}, Number<W2>{}, Number<W3>{})
                    .Fold(I1, Number<K1>{}, Number<K2>{})
                    .Fold(I0, Number<N1>{});

            constexpr auto out_10d_thread_desc =
                fwd(out_k_h_w_n_thread_desc)
                    .Fold(I3, Number<N1>{})
                    .Fold(I2, Number<W1>{}, Number<1>{}, Number<W3>{})
                    .Fold(I0, Number<1>{}, Number<K2>{});

#if 0
            if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
            {
                print_ConstantTensorDescriptor(out_k_h_w_n_thread_desc,
                                               "b: out_k_h_w_n_thread_desc");
                print_ConstantTensorDescriptor(out_10d_thread_desc, "b: out_10d_thread_desc");

                print_ConstantTensorDescriptor(out_n_k_h_w_global_desc,
                                               "b: out_n_k_h_w_global_desc");
                print_ConstantTensorDescriptor(out_10d_global_desc, "b: out_10d_global_desc");
            }
#endif

            constexpr auto map_out_global2thread = Sequence<8, 9, 0, 1, 2, 3, 4, 5, 6, 7>{};

#if 0
            threadwise_tensor_slice_copy_reorder_given_dst2src_v3(
                out_10d_thread_desc,
                p_out_thread,
                out_10d_global_desc,
                p_out_global +
                    out_n_k_h_w_global_desc.GetOffsetFromMultiIndex(
                        n_block_data_begin + n_thread_data_begin,
                        k_block_data_begin + k_thread_data_begin,
                        ho_block_data_begin + ho_thread_data_begin,
                        wo_block_data_begin + wo_thread_data_begin),
                out_10d_thread_desc.GetLengths(),
                map_out_global2thread,
                Number<OutThreadCopyDataPerWrite_W>{});
#else
            threadwise_generic_tensor_slice_copy_v1(
                out_10d_thread_desc.ReorderGivenNew2Old(map_out_global2thread),
                p_out_thread,
                make_zero_array<index_t, 10>(),
                out_10d_global_desc,
                p_out_global +
                    out_n_k_h_w_global_desc.GetOffsetFromMultiIndex(
                        n_block_data_begin + n_thread_data_begin,
                        k_block_data_begin + k_thread_data_begin,
                        ho_block_data_begin + ho_thread_data_begin,
                        wo_block_data_begin + wo_thread_data_begin),
                make_zero_array<index_t, 10>(),
                out_10d_thread_desc.GetLengths().ReorderGivenNew2Old(map_out_global2thread),
                arithmetic_sequence_gen<0, 10, 1>::type{},
                Number<1>{});
#endif
        });
    }
};

} // namespace ck
#endif
