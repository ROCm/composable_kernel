#pragma once
#include "common.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_4d_tensor_op.hpp"
#include "blockwise_2d_tensor_op.hpp"
#include "threadwise_4d_tensor_op.hpp"
#include "blockwise_gemm.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          class LowerPads,
          class UpperPads,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t CPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t NPerThread,
          index_t KPerThread,
          index_t CPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t WeiBlockCopyThreadPerDim0,
          index_t WeiBlockCopyThreadPerDim1>
__global__ void gridwise_implicit_gemm_convolution_1_chwn_cyxk_khwn_padded(
    const Float* const __restrict__ p_in_global,
    const Float* const __restrict__ p_wei_global,
    Float* const __restrict__ p_out_global)
{
    // NPerThread == NPerBlock, because the format of input in LDS [C,Hi,Wi,N]
    //   for GEMM trans([C,K]) * [C,Wo*N], we need a thread to do all the "N"
    // if we use [C,Hi,N,Wi,N] in LDS, then NPerThread can be different from NPerBlock
    static_assert(NPerBlock % NPerThread == 0, "wrong! NPerBlock % NPerThread !=0");
    static_assert((NPerThread < NPerBlock && WoPerThread == 1) || NPerThread == NPerBlock,
                  "wrong!");

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

    constexpr index_t HPadLow = LowerPads{}.Get(I0);
    constexpr index_t WPadLow = LowerPads{}.Get(I1);

    constexpr index_t HPadUp = UpperPads{}.Get(I0);
    constexpr index_t WPadUp = UpperPads{}.Get(I1);

    constexpr index_t HiPerBlock = HoPerBlock + Y - 1;
    constexpr index_t WiPerBlock = WoPerBlock + X - 1;

    // divide block work: [K, Ho, Wo, N]
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

    // flattened (2d) tensor view of wei in global mem
    constexpr auto wei_ek_global_desc = make_ConstantTensorDescriptor(Sequence<C * Y * X, K>{});

    // tensor view of blockwise input and weight in LDS
    constexpr auto in_chwn_block_desc =
        make_ConstantTensorDescriptor(Sequence<CPerBlock, HiPerBlock, WiPerBlock, NPerBlock>{});

    constexpr auto wei_cyxk_block_desc =
        make_ConstantTensorDescriptor(Sequence<CPerBlock, Y, X, KPerBlock>{});

    // flattened (2d) tensor view of wei in LDS
    constexpr auto wei_ek_block_desc =
        make_ConstantTensorDescriptor(Sequence<CPerBlock * Y * X, KPerBlock>{});

    // tensor view of threadwise output in register
    constexpr auto out_hkwn_thread_desc =
        make_ConstantTensorDescriptor(Sequence<HoPerThread, KPerThread, WoPerThread, NPerThread>{});

#if 0
    if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(in_chwn_block_desc, "in_chwn_block_desc");
        print_ConstantTensorDescriptor(wei_cyxk_block_desc, "wei_cyxk_block_desc");
        print_ConstantTensorDescriptor(out_hkwn_thread_desc, "out_hkwn_thread_desc");
    }
#endif

    // blockwise copy
    // input: format is [C, Hi, Wi, N]
    const index_t h_block_pad_low = h_block_work_id == 0 ? HPadLow : 0;
    const index_t w_block_pad_low = w_block_work_id == 0 ? WPadLow : 0;

    const index_t h_block_pad_up = h_block_work_id == HBlockWork - 1 ? HPadUp : 0;
    const index_t w_block_pad_up = w_block_work_id == WBlockWork - 1 ? WPadUp : 0;

#if 0
    if(get_thread_local_1d_id() == 0)
        ;
    {
        printf(
            "%u %u, h_block_pad_low %u w_block_pad_low %u h_block_pad_up %u  w_block_pad_up %u\n",
            get_block_1d_id(),
            get_thread_local_1d_id(),
            h_block_pad_low,
            w_block_pad_low,
            h_block_pad_up,
            w_block_pad_up);
    }
#endif

    constexpr auto blockwise_in_copy =
        BlockwiseChwnTensorCopyPadded<BlockSize,
                                      Float,
                                      decltype(in_chwn_global_desc),
                                      decltype(in_chwn_block_desc),
                                      decltype(in_chwn_block_desc.GetLengths()),
                                      LowerPads>{};

#if 0
    // weight: format is [C,Y,X,K]
    constexpr auto blockwise_wei_copy =
        Blockwise4dTensorCopy1<BlockSize,
                               Float,
                               decltype(wei_cyxk_global_desc),
                               decltype(wei_cyxk_block_desc),
                               decltype(wei_cyxk_block_desc.GetLengths())>{};
#elif 0
    // weight: format is [C*Y*X,K]
    constexpr auto blockwise_wei_copy =
        Blockwise2dTensorCopy1<BlockSize,
                               Float,
                               decltype(wei_ek_global_desc),
                               decltype(wei_ek_block_desc),
                               decltype(wei_ek_block_desc.GetLengths())>{};
#elif 1
    // weight: format is [C*Y*X,K]
    const auto blockwise_wei_copy = Blockwise2dTensorCopy2<BlockSize,
                                                           Float,
                                                           decltype(wei_ek_global_desc),
                                                           decltype(wei_ek_block_desc),
                                                           decltype(wei_ek_block_desc.GetLengths()),
                                                           WeiBlockCopyThreadPerDim0,
                                                           WeiBlockCopyThreadPerDim1>{};
#endif

    // a series of blockwise batched GEMM
    // C_matrix += transpose(A_matrix) * B_matrix
    //   A_matrix and B_matrix saved in LDS, C_matrix saved in register
    //   A_matrix[C,K] is a sub-matrix of wei_block[C,Y,X,K]
    //   B_matrix[C,Wo*N] is a sub-matrix of in_block[C,Hi,Wi,N]
    //   C_matrix[K,Wo*N] is a sub-matrix of out_block[Ho,K,Wo,N]
    constexpr auto a_cxk_block_mtx_desc = make_ConstantMatrixDescriptor(
        Number<CPerBlock>{}, Number<KPerBlock>{}, Number<wei_cyxk_block_desc.GetStride(I0)>{});

    constexpr auto b_cxwn_block_mtx_desc =
        make_ConstantMatrixDescriptor(Number<CPerBlock>{},
                                      Number<WoPerBlock * NPerBlock>{},
                                      Number<in_chwn_block_desc.GetStride(I0)>{});

    constexpr auto c_kxwn_thread_mtx_desc =
        make_ConstantMatrixDescriptor(Number<KPerThread>{}, Number<WoPerThread * NPerThread>{});

    const auto blockwise_batch_gemm =
        Blockwise1dStridedBatchedGemmBlockABlockBThreadC<BlockSize,
                                                         decltype(a_cxk_block_mtx_desc),
                                                         decltype(b_cxwn_block_mtx_desc),
                                                         decltype(c_kxwn_thread_mtx_desc),
                                                         true,
                                                         false,
                                                         false,
                                                         0,
                                                         in_chwn_block_desc.GetStride(I1),
                                                         out_hkwn_thread_desc.GetStride(I0),
                                                         HoPerBlock,
                                                         HoPerThread,
                                                         CPerThread,
                                                         true>{};

    // LDS
    constexpr index_t in_block_element_size  = in_chwn_block_desc.GetElementSpace();
    constexpr index_t wei_block_element_size = wei_cyxk_block_desc.GetElementSpace();

    __shared__ Float p_in_block[in_block_element_size];
    __shared__ Float p_wei_block[wei_block_element_size];

    // register
    Float p_out_thread[out_hkwn_thread_desc.GetElementSpace()];

    // set threadwise output tensor to 0
    threadwise_4d_tensor_set_zero(out_hkwn_thread_desc, p_out_thread);

    const Float* p_wei_global_block_begin =
        p_wei_global + wei_ek_global_desc.GetOffsetFromMultiIndex(0, k_block_data_begin);

    for(index_t c_block_data_begin = 0; c_block_data_begin < C; c_block_data_begin += CPerBlock,
                p_wei_global_block_begin += CPerBlock * wei_ek_global_desc.GetStride(I0),
                __syncthreads())
    {
#if 1
        // input: global mem to LDS,
        blockwise_in_copy.Run(p_in_global,
                              c_block_data_begin,
                              ho_block_data_begin,
                              wo_block_data_begin,
                              n_block_data_begin,
                              p_in_block,
                              h_block_pad_low,
                              w_block_pad_low,
                              h_block_pad_up,
                              w_block_pad_up);
#endif

#if 1
        // weight: global mem to LDS,
        blockwise_wei_copy.Run(p_wei_global_block_begin, p_wei_block);
#endif

        __syncthreads();

        // a series of batched GEMM
        for(index_t y = 0; y < Y; ++y)
        {
            for(index_t x = 0; x < X; ++x)
            {
                auto f_accum = [](auto& acc, const auto&& v) { acc += v; };

                blockwise_batch_gemm.Run(
                    p_wei_block + wei_cyxk_block_desc.GetOffsetFromMultiIndex(0, y, x, 0),
                    p_in_block + in_chwn_block_desc.GetOffsetFromMultiIndex(0, y, x, 0),
                    p_out_thread,
                    f_accum);
            }
        }
    }

    const auto matrix_c_index =
        blockwise_batch_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

    const index_t ho_thread_data_begin = matrix_c_index.batch;
    const index_t k_thread_data_begin  = matrix_c_index.row;
    const index_t wo_thread_data_begin = matrix_c_index.col / NPerBlock;
    const index_t n_thread_data_begin  = matrix_c_index.col - wo_thread_data_begin * NPerBlock;

#if 0
    printf("block %u %u, %u %u %u %u, %u %u %u %u, %f \n", 
            get_block_1d_id(), get_thread_local_1d_id(),
            ho_block_data_begin, k_block_data_begin, wo_block_data_begin, n_block_data_begin,
            ho_thread_data_begin, k_thread_data_begin, wo_thread_data_begin, n_thread_data_begin,
            p_out_thread[0]);
#endif

    // output: register to global mem,
    //   convert out_thread[Ho,K,Wo,N] to out_global[K,Ho,Wo,N]
    constexpr auto reorder_khwn_from_hkwn = Sequence<1, 0, 2, 3>{};

    threadwise_4d_tensor_copy_reorder_by_get_dst_from_src(
        out_hkwn_thread_desc,
        p_out_thread,
        out_khwn_global_desc,
        p_out_global +
            out_khwn_global_desc.GetOffsetFromMultiIndex(k_block_data_begin + k_thread_data_begin,
                                                         ho_block_data_begin + ho_thread_data_begin,
                                                         wo_block_data_begin + wo_thread_data_begin,
                                                         n_block_data_begin + n_thread_data_begin),
        out_hkwn_thread_desc.GetLengths(),
        reorder_khwn_from_hkwn);
}

} // namespace ck
