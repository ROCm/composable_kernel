#pragma once
#include "common.hip.hpp"
#include "ConstantTensorDescriptor.hip.hpp"
#include "ConstantMatrixDescriptor.hip.hpp"
#include "blockwise_2d_tensor_op.hip.hpp"
#include "blockwise_4d_tensor_op.hip.hpp"
#include "threadwise_tensor_slice_op.hip.hpp"
#include "threadwise_4d_tensor_op.hip.hpp"
#include "blockwise_batched_gemm.hip.hpp"

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
          class InBlockCopyClusterLengths_CHWN,
          index_t InBlockCopyDataPerRead_N,
          index_t WeiBlockCopyDataPerRead_K,
          index_t OutThreadCopyDataPerWrite_N>
struct GridwiseConvolutionImplicitGemm_v1r3_lds_double_buffer_chwn_cyxk_khwn
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

        constexpr auto in_c_h_w_n_global_desc  = InGlobalDesc{};
        constexpr auto wei_c_y_x_k_global_desc = WeiGlobalDesc{};
        constexpr auto out_k_h_w_n_global_desc = OutGlobalDesc{};

        constexpr index_t C = in_c_h_w_n_global_desc.GetLength(I0);

        constexpr index_t K  = out_k_h_w_n_global_desc.GetLength(I0);
        constexpr index_t Ho = out_k_h_w_n_global_desc.GetLength(I1);
        constexpr index_t Wo = out_k_h_w_n_global_desc.GetLength(I2);
        constexpr index_t N  = out_k_h_w_n_global_desc.GetLength(I3);

        constexpr index_t Y = wei_c_y_x_k_global_desc.GetLength(I1);
        constexpr index_t X = wei_c_y_x_k_global_desc.GetLength(I2);

        constexpr index_t HiPerBlock = HoPerBlock + Y - 1;
        constexpr index_t WiPerBlock = WoPerBlock + X - 1;

        // assert for LDS double buffer
        static_assert(C % (2 * CPerBlock) == 0, "C cannot be evenly divided");

        // divide block work: [K, Ho, Wo, N]
        static_assert(N % NPerBlock == 0 && K % KPerBlock == 0 && C % CPerBlock == 0 &&
                          Ho % HoPerBlock == 0 && Wo % WoPerBlock == 0,
                      "wrong! cannot evenly divide work for workgroup ");

        constexpr index_t KBlockWork = mod_conv::integer_divide_ceil(K, KPerBlock);
        constexpr index_t HBlockWork = mod_conv::integer_divide_ceil(Ho, HoPerBlock);
        constexpr index_t WBlockWork = mod_conv::integer_divide_ceil(Wo, WoPerBlock);
        constexpr index_t NBlockWork = mod_conv::integer_divide_ceil(N, NPerBlock);

        constexpr auto block_work_desc = make_ConstantTensorDescriptor_packed(
            Sequence<KBlockWork, HBlockWork, WBlockWork, NBlockWork>{});

        const auto block_work_multi_id =
            block_work_desc.GetMultiIndexFrom1dIndex(get_block_1d_id());

        const index_t k_block_data_begin  = block_work_multi_id[0] * KPerBlock;
        const index_t ho_block_data_begin = block_work_multi_id[1] * HoPerBlock;
        const index_t wo_block_data_begin = block_work_multi_id[2] * WoPerBlock;
        const index_t n_block_data_begin  = block_work_multi_id[3] * NPerBlock;

        const index_t hi_block_data_begin = ho_block_data_begin;
        const index_t wi_block_data_begin = wo_block_data_begin;

        // global tensor view
        constexpr auto wei_c_k_global_desc = wei_c_y_x_k_global_desc.Extract(I0, I3);

        // LDS tensor view
        //   be careful of alignment
        constexpr index_t max_align = mod_conv::lcm(InBlockCopyDataPerRead_N,
                                                    WeiBlockCopyDataPerRead_K,
                                                    GemmDataPerReadA,
                                                    GemmDataPerReadB);

        constexpr auto in_c_h_w_n_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, HoPerBlock, WoPerBlock, NPerBlock>{},
            Number<InBlockCopyDataPerRead_N>{});

        // this check is ad-hoc
        // TODO: need to properly implement tensor descriptor with alignment
        static_assert(in_c_h_w_n_block_desc.GetStride(I1) % GemmDataPerReadB == 0,
                      "GemmDataPerReadB alignment requirement is not meet");

        constexpr auto wei_c_k_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, KPerBlock>{},
            Number<mod_conv::lcm(WeiBlockCopyDataPerRead_K, GemmDataPerReadA)>{});

        // tensor view of threadwise output in register
        constexpr auto out_k_h_w_n_thread_desc = make_ConstantTensorDescriptor_packed(
            Sequence<KPerThread, HoPerThread, WoPerThread, NPerThread>{});

// blockwise copy
// input: format is [C, Hi, Wi, N]
#if 0
        const auto blockwise_in_copy =
            Blockwise4dTensorCopy1<BlockSize,
                                   Float,
                                   decltype(in_c_h_w_n_global_desc),
                                   decltype(in_c_h_w_n_block_desc),
                                   decltype(in_c_h_w_n_block_desc.GetLengths()),
                                   InBlockCopyDataPerRead_N>{};
#else
        const auto blockwise_in_copy =
            Blockwise4dTensorCopy3<BlockSize,
                                   Float,
                                   decltype(in_c_h_w_n_global_desc),
                                   decltype(in_c_h_w_n_block_desc),
                                   decltype(in_c_h_w_n_block_desc.GetLengths()),
                                   InBlockCopyClusterLengths_CHWN,
                                   InBlockCopyDataPerRead_N>{};
#endif

        // blockwise wei copy
        //   format is [CPerBlock, X * KPerBlock]
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
            return blockwise_batch_gemm.Run_asm(Xs...);
#else
            return blockwise_batch_gemm.Run_asm_v2(Xs...);
#endif
        };

        // LDS: be careful of alignment
        constexpr index_t in_block_space =
            in_c_h_w_n_block_desc.GetElementSpace(Number<max_align>{});
        constexpr index_t wei_block_space = wei_c_k_block_desc.GetElementSpace(Number<max_align>{});

        // LDS double buffer
        __shared__ Float p_in_block_double[2 * in_block_space];
        __shared__ Float p_wei_block_double[2 * wei_block_space];

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
            print_ConstantTensorDescriptor(wei_c_x_k_block_desc, "wei_c_x_k_block_desc");

            printf("in_block_space %u, wei_block_space %u\n", in_block_space, wei_block_space);
        }
#endif

        // set threadwise output to 0
        threadwise_matrix_set_zero(c_k_wn_thread_mtx_desc, p_out_thread);

        for(index_t y = 0; y < Y; ++y)
        {
            for(index_t x = 0; x < X; ++x)
            {
                const Float* p_in_global_block_offset =
                    p_in_global +
                    in_c_h_w_n_global_desc.GetOffsetFromMultiIndex(
                        0, hi_block_data_begin + y, wi_block_data_begin + x, n_block_data_begin);

                const Float* p_wei_global_block_offset =
                    p_wei_global +
                    wei_c_y_x_k_global_desc.GetOffsetFromMultiIndex(0, y, x, k_block_data_begin);

                // LDS double buffer: preload data into LDS
                {
                    Float p_in_register_clipboard[blockwise_in_copy.GetRegisterClipboardSize()];
                    Float p_wei_register_clipboard[blockwise_wei_copy.GetRegisterClipboardSize()];

                    blockwise_in_copy.RunLoadRegisterClipboard(p_in_global_block_offset,
                                                               p_in_register_clipboard);
                    blockwise_wei_copy.RunLoadRegisterClipboard(p_wei_global_block_offset,
                                                                p_wei_register_clipboard);

                    blockwise_in_copy.RunStoreRegisterClipboard(p_in_register_clipboard,
                                                                p_in_block_double);
                    blockwise_wei_copy.RunStoreRegisterClipboard(p_wei_register_clipboard,
                                                                 p_wei_block_double);
                }

                // LDS double buffer: main body
                for(index_t c_block_data_begin = 0; c_block_data_begin + 2 * CPerBlock < C;
                    c_block_data_begin += 2 * CPerBlock)
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
                        Float
                            p_wei_register_clipboard[blockwise_wei_copy.GetRegisterClipboardSize()];

                        p_in_global_block_offset +=
                            CPerBlock * in_c_h_w_n_global_desc.GetStride(I0);
                        p_wei_global_block_offset +=
                            CPerBlock * wei_c_y_x_k_global_desc.GetStride(I0);

                        __syncthreads();

                        // LDS doubel buffer: load next data from device mem
                        blockwise_in_copy.RunLoadRegisterClipboard(p_in_global_block_offset,
                                                                   p_in_register_clipboard);
                        blockwise_wei_copy.RunLoadRegisterClipboard(p_wei_global_block_offset,
                                                                    p_wei_register_clipboard);

                        run_blockwise_batch_gemm(p_wei_block_now, p_in_block_now, p_out_thread);

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
                    p_in_global_block_offset += CPerBlock * in_c_h_w_n_global_desc.GetStride(I0);
                    p_wei_global_block_offset += CPerBlock * wei_c_y_x_k_global_desc.GetStride(I0);

                    __syncthreads();

                    // LDS doubel buffer: load next data from device mem
                    blockwise_in_copy.RunLoadRegisterClipboard(p_in_global_block_offset,
                                                               p_in_register_clipboard);
                    blockwise_wei_copy.RunLoadRegisterClipboard(p_wei_global_block_offset,
                                                                p_wei_register_clipboard);

                    // LDS double buffer: GEMM on current data
                    run_blockwise_batch_gemm(p_wei_block_double, p_in_block_double, p_out_thread);

                    // LDS double buffer: store next data to LDS
                    blockwise_in_copy.RunStoreRegisterClipboard(p_in_register_clipboard,
                                                                p_in_block_double + in_block_space);
                    blockwise_wei_copy.RunStoreRegisterClipboard(
                        p_wei_register_clipboard, p_wei_block_double + wei_block_space);

                    // odd iteration
                    __syncthreads();

                    // LDS double buffer: GEMM on current data
                    run_blockwise_batch_gemm(p_wei_block_double + wei_block_space,
                                             p_in_block_double + in_block_space,
                                             p_out_thread);
                }
            }
        }

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
            // being instantiated here
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

            constexpr auto out_10d_global_desc = fwd(out_k_h_w_n_global_desc)
                                                     .Fold(I3, Number<N1>{}, Number<N2>{})
                                                     .Fold(I2, Number<W1>{}, Number<W2>{})
                                                     .Fold(I0, Number<K1>{}, Number<K2>{});

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

                print_ConstantTensorDescriptor(out_k_h_w_n_global_desc,
                                               "a: out_k_h_w_n_global_desc");
                print_ConstantTensorDescriptor(out_10d_global_desc, "a: out_10d_global_desc");
            }
#endif

            threadwise_tensor_slice_copy(out_10d_thread_desc,
                                         p_out_thread,
                                         out_10d_global_desc,
                                         p_out_global +
                                             out_k_h_w_n_global_desc.GetOffsetFromMultiIndex(
                                                 k_block_data_begin + k_thread_data_begin,
                                                 ho_block_data_begin + ho_thread_data_begin,
                                                 wo_block_data_begin + wo_thread_data_begin,
                                                 n_block_data_begin + n_thread_data_begin),
                                         out_10d_thread_desc.GetLengths(),
                                         Number<OutThreadCopyDataPerWrite_N>{});
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
                fwd(out_k_h_w_n_global_desc)
                    .Fold(I3, Number<N1>{})
                    .Fold(I2, Number<W1>{}, Number<W2>{}, Number<W3>{})
                    .Fold(I0, Number<K1>{}, Number<K2>{});

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

                print_ConstantTensorDescriptor(out_k_h_w_n_global_desc,
                                               "b: out_k_h_w_n_global_desc");
                print_ConstantTensorDescriptor(out_10d_global_desc, "b: out_10d_global_desc");
            }
#endif

            threadwise_tensor_slice_copy(out_10d_thread_desc,
                                         p_out_thread,
                                         out_10d_global_desc,
                                         p_out_global +
                                             out_k_h_w_n_global_desc.GetOffsetFromMultiIndex(
                                                 k_block_data_begin + k_thread_data_begin,
                                                 ho_block_data_begin + ho_thread_data_begin,
                                                 wo_block_data_begin + wo_thread_data_begin,
                                                 n_block_data_begin + n_thread_data_begin),
                                         out_10d_thread_desc.GetLengths(),
                                         Number<OutThreadCopyDataPerWrite_N>{});
        });
    }
};
