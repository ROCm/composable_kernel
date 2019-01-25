#pragma once
#include "common.cuh"
#include "ConstantTensorDescriptor.cuh"
#include "ConstantMatrixDescriptor.cuh"
#include "blockwise_4d_tensor_op.cuh"
#include "threadwise_4d_tensor_op.cuh"
#include "gemm.cuh"

template <unsigned GridSize,
          unsigned BlockSize,
          class Float,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          unsigned NPerBlock,
          unsigned KPerBlock,
          unsigned CPerBlock,
          unsigned HoPerBlock,
          unsigned WoPerBlock,
          unsigned NPerThread,
          unsigned KPerThread,
          unsigned CPerThread,
          unsigned HoPerThread,
          unsigned WoPerThread>
__global__ void
gridwise_implicit_gemm_convolution_1_chwn_csrk_khwn(InGlobalDesc,
                                                    Float* const __restrict__ p_in_global,
                                                    WeiGlobalDesc,
                                                    Float* const __restrict__ p_wei_global,
                                                    OutGlobalDesc,
                                                    Float* __restrict__ p_out_global)
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
    constexpr auto wei_csrk_global_desc = WeiGlobalDesc{};
    constexpr auto out_khwn_global_desc = OutGlobalDesc{};

    constexpr unsigned C = in_chwn_global_desc.GetLength(I0);

    constexpr unsigned K  = out_khwn_global_desc.GetLength(I0);
    constexpr unsigned Ho = out_khwn_global_desc.GetLength(I1);
    constexpr unsigned Wo = out_khwn_global_desc.GetLength(I2);
    constexpr unsigned N  = out_khwn_global_desc.GetLength(I3);

    constexpr unsigned S = wei_csrk_global_desc.GetLength(I1);
    constexpr unsigned R = wei_csrk_global_desc.GetLength(I2);

    constexpr unsigned HiPerBlock = HoPerBlock + S - 1;
    constexpr unsigned WiPerBlock = WoPerBlock + R - 1;

    // divide block work: [K, Ho, Wo, N]
    constexpr unsigned KBlockWork = (K + KPerBlock - 1) / KPerBlock;
    constexpr unsigned HBlockWork = (Ho + HoPerBlock - 1) / HoPerBlock;
    constexpr unsigned WBlockWork = (Wo + WoPerBlock - 1) / WoPerBlock;
    constexpr unsigned NBlockWork = (N + NPerBlock - 1) / NPerBlock;

    const unsigned k_block_work_id = get_block_1d_id() / (HBlockWork * WBlockWork * NBlockWork);
    unsigned itmp = get_block_1d_id() - k_block_work_id * (HBlockWork * WBlockWork * NBlockWork);
    const unsigned h_block_work_id = itmp / (WBlockWork * NBlockWork);
    itmp -= h_block_work_id * (WBlockWork * NBlockWork);
    const unsigned w_block_work_id = itmp / NBlockWork;
    const unsigned n_block_work_id = itmp - w_block_work_id * NBlockWork;

    const unsigned k_block_data_begin  = k_block_work_id * KPerBlock;
    const unsigned ho_block_data_begin = h_block_work_id * HoPerBlock;
    const unsigned wo_block_data_begin = w_block_work_id * WoPerBlock;
    const unsigned n_block_data_begin  = n_block_work_id * NPerBlock;

    const unsigned hi_block_data_begin = ho_block_data_begin;
    const unsigned wi_block_data_begin = wo_block_data_begin;

    // tensor view of blockwise input and weight in LDS
    constexpr auto in_chwn_block_desc =
        make_ConstantTensorDescriptor(Sequence<CPerBlock, HiPerBlock, WiPerBlock, NPerBlock>{});

    constexpr auto wei_csrk_block_desc =
        make_ConstantTensorDescriptor(Sequence<CPerBlock, S, R, KPerBlock>{});

    // tensor view of threadwise output in register
    constexpr auto out_hkwn_thread_desc =
        make_ConstantTensorDescriptor(Sequence<HoPerThread, KPerThread, WoPerThread, NPerThread>{});

#if 0
    if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(in_nchw_block_desc, "in_nchw_block_desc");
        print_ConstantTensorDescriptor(in_chwn_block_desc, "in_chwn_block_desc");

        print_ConstantTensorDescriptor(wei_srck_block_desc, "wei_srck_block_desc");

        print_ConstantTensorDescriptor(out_hkwn_thread_desc, "out_hkwn_thread_desc");
    }
#endif

    // blockwise copy
    // input: format is [C, Hi, Wi, N]
    constexpr auto blockwise_in_copy =
        blockwise_4d_tensor_copy_1<BlockSize,
                                   Float,
                                   decltype(in_chwn_global_desc),
                                   decltype(in_chwn_block_desc),
                                   decltype(in_chwn_block_desc.GetLengths())>{};

    // weight: format is [S,R,C,K]
    constexpr auto blockwise_wei_copy =
        blockwise_4d_tensor_copy_1<BlockSize,
                                   Float,
                                   decltype(wei_csrk_global_desc),
                                   decltype(wei_csrk_block_desc),
                                   decltype(wei_csrk_block_desc.GetLengths())>{};

    // a series of blockwise batched GEMM
    // C_matrix += transpose(A_matrix) * B_matrix
    //   A_matrix and B_matrix saved in LDS, C_matrix saved in register
    //   A_matrix[C,K] is a sub-matrix of wei_block[S,R,C,K]
    //   B_matrix[C,Wo*N] is a sub-matrix of in_block[C,Hi,Wi,N]
    //   C_matrix[K,Wo*N] is a sub-matrix of out_block[Ho,K,Wo,N]
    const auto a_cxk_block_mtx_desc = make_ConstantMatrixDescriptor(
        Number<CPerBlock>{},
        Number<KPerBlock>{},
        Number<wei_csrk_block_desc.GetStride(I0)>{}); // constexpr doesn't compile

    const auto b_cxwn_block_mtx_desc = make_ConstantMatrixDescriptor(
        Number<CPerBlock>{},
        Number<WoPerBlock * NPerBlock>{},
        Number<in_chwn_block_desc.GetStride(I0)>{}); // constexpr doesn't compile

    const auto c_kxwn_thread_mtx_desc = make_ConstantMatrixDescriptor(
        Number<KPerThread>{}, Number<WoPerThread * NPerThread>{}); // constexpr doesn't compile

    const auto blockwise_batch_gemm =
        blockwise_1d_strided_batched_gemm_block_a_block_b_thread_c<BlockSize,
                                                                   decltype(a_cxk_block_mtx_desc),
                                                                   decltype(b_cxwn_block_mtx_desc),
                                                                   decltype(c_kxwn_thread_mtx_desc),
                                                                   true,
                                                                   false,
                                                                   false,
                                                                   0,
                                                                   in_chwn_block_desc.GetStride(I1),
                                                                   out_hkwn_thread_desc.GetStride(
                                                                       I0),
                                                                   HoPerBlock,
                                                                   HoPerThread,
                                                                   CPerThread,
                                                                   true>{};

    // LDS
    constexpr unsigned in_block_size  = in_chwn_block_desc.GetElementSpace();
    constexpr unsigned wei_block_size = wei_csrk_block_desc.GetElementSpace();

    __shared__ Float p_in_block[in_block_size];
    __shared__ Float p_wei_block[wei_block_size];

    // register
    Float p_out_thread[out_hkwn_thread_desc.GetElementSpace()];

    // set threadwise output tensor to 0
    threadwise_4d_tensor_set_zero(out_hkwn_thread_desc, p_out_thread);

    for(unsigned c_block_data_begin = 0; c_block_data_begin < C;
        c_block_data_begin += CPerBlock, __syncthreads())
    {
#if 1
        // input: global mem to LDS,
        blockwise_in_copy.run(p_in_global + in_chwn_global_desc.Get1dIndex(c_block_data_begin,
                                                                           hi_block_data_begin,
                                                                           wi_block_data_begin,
                                                                           n_block_data_begin),
                              p_in_block);
#endif

#if 1
        // weight: global mem to LDS,
        blockwise_wei_copy.run(p_wei_global + wei_csrk_global_desc.Get1dIndex(
                                                  c_block_data_begin, 0, 0, k_block_data_begin),
                               p_wei_block);
#endif

        __syncthreads();

        // a series of batched GEMM
        for(unsigned s = 0; s < S; ++s)
        {
            for(unsigned r = 0; r < R; ++r)
            {
                auto f_accum = [](auto& acc, const auto&& v) { acc += v; };

                blockwise_batch_gemm.run(p_wei_block + wei_csrk_block_desc.Get1dIndex(0, s, r, 0),
                                         p_in_block + in_chwn_block_desc.Get1dIndex(0, s, r, 0),
                                         p_out_thread,
                                         f_accum);
            }
        }
    }

    const auto matrix_c_index =
        blockwise_batch_gemm.CalculateThreadMatrixCIndex(get_thread_local_1d_id());

    const unsigned ho_thread_data_begin = matrix_c_index.batch_begin;
    const unsigned k_thread_data_begin  = matrix_c_index.row_begin;
    const unsigned wo_thread_data_begin = matrix_c_index.col_begin / NPerBlock;
    const unsigned n_thread_data_begin =
        matrix_c_index.col_begin - wo_thread_data_begin * NPerBlock;

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
        p_out_global + out_khwn_global_desc.Get1dIndex(k_block_data_begin + k_thread_data_begin,
                                                       ho_block_data_begin + ho_thread_data_begin,
                                                       wo_block_data_begin + wo_thread_data_begin,
                                                       n_block_data_begin + n_thread_data_begin),
        out_hkwn_thread_desc.GetLengths(),
        reorder_khwn_from_hkwn);
}
