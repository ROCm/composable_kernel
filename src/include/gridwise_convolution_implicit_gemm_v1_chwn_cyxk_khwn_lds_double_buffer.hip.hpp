#pragma once
#include "common.hip.hpp"
#include "ConstantTensorDescriptor.hip.hpp"
#include "ConstantMatrixDescriptor.hip.hpp"
#include "blockwise_4d_tensor_op.hip.hpp"
#include "blockwise_2d_tensor_op.hip.hpp"
#include "threadwise_nd_tensor_op.hip.hpp"
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
          class InBlockCopyThreadPerDims,
          index_t InBlockCopyDataPerRead,
          index_t WeiBlockCopyDataPerRead,
          index_t OutThreadCopyDataPerWrite>
struct GridwiseConvolutionImplicitGemm_v1_chwn_cyxk_khwn_lds_double_buffer
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        // be careful of this assertion
        static_assert(
            NPerThread <= NPerBlock && NPerBlock % NPerThread == 0,
            "wrong! should satisfy: NPerThread <= NPerBlock && NPerBlock % NPerThread == 0");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto in_chwn_global_desc  = InGlobalDesc{};
        constexpr auto wei_cyxk_global_desc = WeiGlobalDesc{};
        constexpr auto out_khwn_global_desc = OutGlobalDesc{};

        constexpr index_t C = in_chwn_global_desc.GetLength(I0);

        constexpr index_t K  = out_khwn_global_desc.GetLength(I0);
        constexpr index_t Ho = out_khwn_global_desc.GetLength(I1);
        constexpr index_t Wo = out_khwn_global_desc.GetLength(I2);
        constexpr index_t N  = out_khwn_global_desc.GetLength(I3);

        constexpr index_t Y = wei_cyxk_global_desc.GetLength(I1);
        constexpr index_t X = wei_cyxk_global_desc.GetLength(I2);

        constexpr index_t HiPerBlock = HoPerBlock + Y - 1;
        constexpr index_t WiPerBlock = WoPerBlock + X - 1;

        // assert for LDS double buffer
        static_assert(C % (2 * CPerBlock) == 0, "C cannot be evenly divided");

        // divide block work: [K, Ho, Wo, N]
        static_assert(N % NPerBlock == 0 && K % KPerBlock == 0 && C % CPerBlock == 0 &&
                          Ho % HoPerBlock == 0 && Wo % WoPerBlock == 0,
                      "wrong! cannot evenly divide work for workgroup ");

        constexpr index_t KBlockWork = (K + KPerBlock - 1) / KPerBlock;
        constexpr index_t HBlockWork = (Ho + HoPerBlock - 1) / HoPerBlock;
        constexpr index_t WBlockWork = (Wo + WoPerBlock - 1) / WoPerBlock;
        constexpr index_t NBlockWork = (N + NPerBlock - 1) / NPerBlock;

        const index_t k_block_work_id = get_block_1d_id() / (HBlockWork * WBlockWork * NBlockWork);
        index_t itmp = get_block_1d_id() - k_block_work_id * (HBlockWork * WBlockWork * NBlockWork);
        const index_t h_block_work_id = itmp / (WBlockWork * NBlockWork);
        itmp -= h_block_work_id * (WBlockWork * NBlockWork);
        const index_t w_block_work_id = itmp / NBlockWork;
        const index_t n_block_work_id = itmp - w_block_work_id * NBlockWork;

        const index_t k_block_data_begin  = k_block_work_id * KPerBlock;
        const index_t ho_block_data_begin = h_block_work_id * HoPerBlock;
        const index_t wo_block_data_begin = w_block_work_id * WoPerBlock;
        const index_t n_block_data_begin  = n_block_work_id * NPerBlock;

        const index_t hi_block_data_begin = ho_block_data_begin;
        const index_t wi_block_data_begin = wo_block_data_begin;

        // flattend (2d) tensor view of gridwise weight
        constexpr auto wei_ek_global_desc = make_ConstantTensorDescriptor(Sequence<C * Y * X, K>{});

        // tensor view of blockwise input and weight in LDS
        //   be careful of alignment
        constexpr index_t max_align =
            mod_conv::max(index_t(4), InBlockCopyDataPerRead, WeiBlockCopyDataPerRead);

        constexpr auto in_chwn_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, HiPerBlock, WiPerBlock, NPerBlock>{}, Number<max_align>{});

        constexpr auto wei_ek_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock * Y * X, KPerBlock>{}, Number<max_align>{});

        constexpr auto wei_cyxk_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, Y, X, KPerBlock>{}, Number<max_align>{});

        // tensor view of threadwise output in register
        constexpr auto out_khwn_thread_desc = make_ConstantTensorDescriptor(
            Sequence<KPerThread, HoPerThread, WoPerThread, NPerThread>{});

        // blockwise copy
        // input: format is [C, Hi, Wi, N]
        const auto blockwise_in_copy =
            Blockwise4dTensorCopy3<BlockSize,
                                   Float,
                                   decltype(in_chwn_global_desc),
                                   decltype(in_chwn_block_desc),
                                   decltype(in_chwn_block_desc.GetLengths()),
                                   InBlockCopyThreadPerDims,
                                   InBlockCopyDataPerRead>{};

        // blockwise wei copy
        //   format is [CPerBlock*Y*X,KPerBlock]
        const auto blockwise_wei_copy =
            Blockwise2dTensorCopy3<BlockSize,
                                   Float,
                                   decltype(wei_ek_global_desc),
                                   decltype(wei_ek_block_desc),
                                   decltype(wei_ek_block_desc.GetLengths()),
                                   WeiBlockCopyDataPerRead>{};

        // a series of blockwise batched GEMM
        // C_matrix += transpose(A_matrix) * B_matrix
        //   A_matrix and B_matrix saved in LDS, C_matrix saved in register
        //   A_matrix[C,K] is a sub-matrix of wei_block[C,Y,X,K]
        //   B_matrix[C,Wo*N] is a sub-matrix of in_block[C,Hi,Wi,N]
        //   C_matrix[K,Wo*N] is a sub-matrix of out_block[K,Ho,Wo,N]
        constexpr auto a_cxk_block_mtx_desc = make_ConstantMatrixDescriptor(
            Number<CPerBlock>{}, Number<KPerBlock>{}, Number<wei_cyxk_block_desc.GetStride(I0)>{});

        constexpr auto b_cxwn_block_mtx_desc =
            make_ConstantMatrixDescriptor(Number<CPerBlock>{},
                                          Number<WoPerBlock * NPerBlock>{},
                                          Number<in_chwn_block_desc.GetStride(I0)>{});

        constexpr auto c_kxwn_thread_mtx_desc =
            make_ConstantMatrixDescriptor(Number<KPerThread>{},
                                          Number<WoPerThread * NPerThread>{},
                                          Number<out_khwn_thread_desc.GetStride(I0)>{});

        const auto blockwise_batch_gemm =
            BlockwiseBatchGemmBlockABlockBThreadCTransANormalBNormalC_V2<
                BlockSize,
                decltype(a_cxk_block_mtx_desc),
                decltype(b_cxwn_block_mtx_desc),
                decltype(c_kxwn_thread_mtx_desc),
                0,
                in_chwn_block_desc.GetStride(I1),
                out_khwn_thread_desc.GetStride(I1),
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

        // LDS: be careful of alignment
        constexpr index_t in_block_space = in_chwn_block_desc.GetElementSpace(Number<max_align>{});

        constexpr index_t wei_block_space =
            wei_cyxk_block_desc.GetElementSpace(Number<max_align>{});

        // LDS double buffer
        __shared__ Float p_in_block_double[2 * in_block_space];
        __shared__ Float p_wei_block_double[2 * wei_block_space];

        const Float* p_in_global_block_offset =
            p_in_global +
            in_chwn_global_desc.Get1dIndex(
                0, hi_block_data_begin, wi_block_data_begin, n_block_data_begin);

        const Float* p_wei_global_block_offset =
            p_wei_global + wei_cyxk_global_desc.Get1dIndex(0, 0, 0, k_block_data_begin);

        // preload data into LDS
        {
            Float p_in_register_clipboard[blockwise_in_copy.GetRegisterClipboardSize()];
            Float p_wei_register_clipboard[blockwise_wei_copy.GetRegisterClipboardSize()];

            blockwise_in_copy.RunLoadRegisterClipboard(p_in_global_block_offset,
                                                       p_in_register_clipboard);
            blockwise_wei_copy.RunLoadRegisterClipboard(p_wei_global_block_offset,
                                                        p_wei_register_clipboard);

            blockwise_in_copy.RunStoreRegisterClipboard(p_in_register_clipboard, p_in_block_double);
            blockwise_wei_copy.RunStoreRegisterClipboard(p_wei_register_clipboard,
                                                         p_wei_block_double);
        }

        // register
        Float p_out_thread[out_khwn_thread_desc.GetElementSpace()];

        // set threadwise output tensor to 0
        threadwise_4d_tensor_set_zero(out_khwn_thread_desc, p_out_thread);

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

                // load next data
                Float p_in_register_clipboard[blockwise_in_copy.GetRegisterClipboardSize()];
                Float p_wei_register_clipboard[blockwise_wei_copy.GetRegisterClipboardSize()];

                p_in_global_block_offset += CPerBlock * in_chwn_global_desc.GetStride(I0);
                p_wei_global_block_offset += CPerBlock * wei_cyxk_global_desc.GetStride(I0);

                __syncthreads();

                blockwise_in_copy.RunLoadRegisterClipboard(p_in_global_block_offset,
                                                           p_in_register_clipboard);

                blockwise_wei_copy.RunLoadRegisterClipboard(p_wei_global_block_offset,
                                                            p_wei_register_clipboard);

                // a series of batched GEMM
                for(index_t y = 0; y < Y; ++y)
                {
                    for(index_t x = 0; x < X; ++x)
                    {
                        blockwise_batch_gemm.Run(
                            p_wei_block_now + wei_cyxk_block_desc.Get1dIndex(0, y, x, 0),
                            p_in_block_now + in_chwn_block_desc.Get1dIndex(0, y, x, 0),
                            p_out_thread);
                    }
                }

                blockwise_in_copy.RunStoreRegisterClipboard(p_in_register_clipboard,
                                                            p_in_block_next);
                blockwise_wei_copy.RunStoreRegisterClipboard(p_wei_register_clipboard,
                                                             p_wei_block_next);
            }
        }

        // tail
        {
            // even
            p_in_global_block_offset += CPerBlock * in_chwn_global_desc.GetStride(I0);
            p_wei_global_block_offset += CPerBlock * wei_cyxk_global_desc.GetStride(I0);

            __syncthreads();

            Float p_in_register_clipboard[blockwise_in_copy.GetRegisterClipboardSize()];
            Float p_wei_register_clipboard[blockwise_wei_copy.GetRegisterClipboardSize()];

            blockwise_in_copy.RunLoadRegisterClipboard(p_in_global_block_offset,
                                                       p_in_register_clipboard);

            blockwise_wei_copy.RunLoadRegisterClipboard(p_wei_global_block_offset,
                                                        p_wei_register_clipboard);

            for(index_t y = 0; y < Y; ++y)
            {
                for(index_t x = 0; x < X; ++x)
                {
                    blockwise_batch_gemm.Run(
                        p_wei_block_double + wei_cyxk_block_desc.Get1dIndex(0, y, x, 0),
                        p_in_block_double + in_chwn_block_desc.Get1dIndex(0, y, x, 0),
                        p_out_thread);
                }
            }

            blockwise_in_copy.RunStoreRegisterClipboard(p_in_register_clipboard,
                                                        p_in_block_double + in_block_space);

            blockwise_wei_copy.RunStoreRegisterClipboard(p_wei_register_clipboard,
                                                         p_wei_block_double + wei_block_space);

            // odd
            __syncthreads();

            for(index_t y = 0; y < Y; ++y)
            {
                for(index_t x = 0; x < X; ++x)
                {
                    blockwise_batch_gemm.Run(p_wei_block_double + wei_block_space +
                                                 wei_cyxk_block_desc.Get1dIndex(0, y, x, 0),
                                             p_in_block_double + in_block_space +
                                                 in_chwn_block_desc.Get1dIndex(0, y, x, 0),
                                             p_out_thread);
                }
            }
        }

// output: register to global mem,
#if 0
        const auto c_thread_mtx_begin =
            blockwise_batch_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        for(index_t k = 0; k < out_khwn_thread_desc.GetLength(I0); ++k)
        {
            for(index_t ho = 0; ho < out_khwn_thread_desc.GetLength(I1); ++ho)
            {
                for(index_t wo = 0; wo < out_khwn_thread_desc.GetLength(I2); ++wo)
                {
                    for(index_t n = 0; n < out_khwn_thread_desc.GetLength(I3); ++n)
                    {
                        const index_t b = out_khwn_thread_desc.Get1dIndex(0, 0, wo, n);

                        const auto c_thread_mtx_distance =
                            blockwise_batch_gemm.GetDistanceFromBeginOfThreadMatrixC(ho, k, b);

                        const index_t ho_thread =
                            c_thread_mtx_begin.batch + c_thread_mtx_distance.batch;
                        const index_t k_thread = c_thread_mtx_begin.row + c_thread_mtx_distance.row;
                        const index_t b_thread = c_thread_mtx_begin.col + c_thread_mtx_distance.col;

                        const index_t wo_thread = b_thread / NPerBlock;
                        const index_t n_thread  = b_thread % NPerBlock;

                        p_out_global[out_khwn_global_desc.Get1dIndex(k_block_data_begin + k_thread,
                                                                     ho_block_data_begin + ho_thread,
                                                                     wo_block_data_begin + wo_thread,
                                                                     n_block_data_begin + n_thread)] =
                            p_out_thread[out_khwn_thread_desc.Get1dIndex(k, ho, wo, n)];
                    }
                }
            }
        }
#elif 1
        const auto c_thread_mtx_begin =
            blockwise_batch_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        const index_t k_thread_data_begin  = c_thread_mtx_begin.row;
        const index_t ho_thread_data_begin = c_thread_mtx_begin.batch;
        const index_t wo_thread_data_begin = c_thread_mtx_begin.col / NPerBlock;
        const index_t n_thread_data_begin =
            c_thread_mtx_begin.col - NPerBlock * wo_thread_data_begin;

        // output is a 10d tensor
        constexpr index_t N2 = GemmNPerThreadSubC;
        constexpr index_t N1 = NPerBlock / N2;

        constexpr index_t W2 =
            (GemmNLevel0Cluster * GemmNLevel1Cluster) / (NPerBlock / GemmNPerThreadSubC);
        constexpr index_t W1 = WoPerBlock / W2;

        constexpr index_t K2 = GemmMPerThreadSubC;
        constexpr index_t K1 = KPerBlock / KPerThread;

        constexpr auto out_10d_global_desc = make_ConstantTensorDescriptor(
            Sequence<K / (K1 * K2), K1, K2, Ho, Wo / (W1 * W2), W1, W2, N / (N1 * N2), N1, N2>{});

        constexpr auto out_10d_thread_desc = make_ConstantTensorDescriptor(
            Sequence<KPerThread / K2, 1, K2, HoPerThread, 1, W1, 1, 1, 1, N2>{});

#if 0
        if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
        {
            print_ConstantTensorDescriptor(out_khwn_thread_desc, "out_khwn_thread_desc");
            print_ConstantTensorDescriptor(out_10d_thread_desc, "out_10d_thread_desc");

            print_ConstantTensorDescriptor(out_khwn_global_desc, "out_khwn_global_desc");
            print_ConstantTensorDescriptor(out_10d_global_desc, "out_10d_global_desc");
        }
#endif

        threadwise_10d_tensor_copy(
            out_10d_thread_desc,
            p_out_thread,
            out_10d_global_desc,
            p_out_global +
                out_khwn_global_desc.Get1dIndex(k_block_data_begin + k_thread_data_begin,
                                                ho_block_data_begin + ho_thread_data_begin,
                                                wo_block_data_begin + wo_thread_data_begin,
                                                n_block_data_begin + n_thread_data_begin),
            out_10d_thread_desc.GetLengths(),
            Number<OutThreadCopyDataPerWrite>{});
#endif
    }
};
