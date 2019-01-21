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
          unsigned KPerThread,
          unsigned CPerThread,
          unsigned HoPerThread,
          unsigned WoPerThread>
__global__ void
gridwise_implicit_gemm_convolution_1_nchw_kcsr(InGlobalDesc,
                                               Float* const __restrict__ p_in_global,
                                               WeiGlobalDesc,
                                               Float* const __restrict__ p_wei_global,
                                               OutGlobalDesc,
                                               Float* __restrict__ p_out_global)
{
    // NPerThread == NPerBlock, because the format of input in LDS [C,Hi,Wi,N]
    //   for GEMM trans([C,K]) * [C,Wo*N], we need a thread to do all the "N"
    // if we use [C,Hi,N,Wi,N] in LDS, then NPerThread can be different from NPerBlock
    constexpr unsigned NPerThread = NPerBlock;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_global_desc  = InGlobalDesc{};
    constexpr auto wei_kcsr_global_desc = WeiGlobalDesc{};
    constexpr auto out_nkhw_global_desc = OutGlobalDesc{};

    constexpr unsigned S = wei_kcsr_global_desc.GetLength(I2);
    constexpr unsigned R = wei_kcsr_global_desc.GetLength(I3);

    constexpr unsigned HiPerBlock = HoPerBlock + S - 1;
    constexpr unsigned WiPerBlock = WoPerBlock + R - 1;

    // divide block work: NCHW
    constexpr unsigned NBlockWork =
        (out_nkhw_global_desc.GetLength(I0) + NPerBlock - 1) / NPerBlock;
    constexpr unsigned KBlockWork =
        (out_nkhw_global_desc.GetLength(I1) + KPerBlock - 1) / KPerBlock;
    constexpr unsigned HBlockWork =
        (out_nkhw_global_desc.GetLength(I2) + HoPerBlock - 1) / HoPerBlock;
    constexpr unsigned WBlockWork =
        (out_nkhw_global_desc.GetLength(I3) + WoPerBlock - 1) / WoPerBlock;

    // tensor view of un-reorderd blockwise input and weight (imaginary)
    constexpr auto in_nchw_block_desc =
        make_ConstantTensorDescriptor(Sequence<NPerBlock, CPerBlock, HiPerBlock, WiPerBlock>{});

    constexpr auto wei_kcsr_block_desc =
        make_ConstantTensorDescriptor(Sequence<KPerBlock, CPerBlock, S, R>{});

    // tensor view of reordered blockwise input and weight in LDS
    constexpr auto reorder_srck_from_kcsr = Sequence<2, 3, 1, 0>{};
    constexpr auto wei_srck_block_desc    = make_ConstantTensorDescriptor(
        wei_kcsr_block_desc.GetLengths().ReorderByGetNewFromOld(reorder_srck_from_kcsr));

    constexpr auto reorder_chwn_from_nchw = Sequence<1, 2, 3, 0>{};
    constexpr auto in_chwn_block_desc     = make_ConstantTensorDescriptor(
        in_nchw_block_desc.GetLengths().ReorderByGetNewFromOld(reorder_chwn_from_nchw));

    // tensor view of threadwise output in register
    constexpr auto out_hkwn_thread_desc =
        make_ConstantTensorDescriptor(Sequence<HoPerThread, KPerThread, WoPerThread, NPerThread>{});

#if 0
    if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(in_nchw_block_desc, "in_nchw_block_desc");
        print_ConstantTensorDescriptor(in_chwn_block_desc, "in_chwn_block_desc");

        print_ConstantTensorDescriptor(wei_kcsr_block_desc, "wei_kcsr_block_desc");
        print_ConstantTensorDescriptor(wei_srck_block_desc, "wei_srck_block_desc");

        print_ConstantTensorDescriptor(out_hkwn_thread_desc, "out_hkwn_thread_desc");
    }
#endif

    // my block work
    unsigned itmp                  = get_block_1d_id();
    const unsigned n_block_work_id = itmp / (KBlockWork * HBlockWork * WBlockWork);
    itmp -= n_block_work_id * (KBlockWork * HBlockWork * WBlockWork);
    const unsigned k_block_work_id = itmp / (HBlockWork * WBlockWork);
    itmp -= k_block_work_id * (HBlockWork * WBlockWork);
    const unsigned h_block_work_id = itmp / WBlockWork;
    const unsigned w_block_work_id = itmp - h_block_work_id * WBlockWork;

    const unsigned n_block_data_begin  = n_block_work_id * NPerBlock;
    const unsigned k_block_data_begin  = k_block_work_id * KPerBlock;
    const unsigned ho_block_data_begin = h_block_work_id * HoPerBlock;
    const unsigned wo_block_data_begin = w_block_work_id * HoPerBlock;

    const unsigned hi_block_data_begin = ho_block_data_begin;
    const unsigned wi_block_data_begin = wo_block_data_begin;

    // a series of blockwise batched GEMM
    // C_matrix += transpose(A_matrix) * B_matrix
    //   A_matrix and B_matrix saved in LDS, C_matrix saved in register
    //   A_matrix[C,K] is a sub-matrix of wei_block[S,R,C,K]
    //   B_matrix[C,Wo*N] is a sub-matrix of in_block[C,Hi,Wi,N]
    //   C_matrix[K,Wo*N] is a sub-matrix of out_block[Ho,K,Wo,N]
    const auto a_cxk_block_mtx_desc = make_ConstantMatrixDescriptor(
        Number<CPerBlock>{}, Number<KPerBlock>{}); // constexpr doesn't compile

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
    constexpr unsigned wei_block_size = wei_srck_block_desc.GetElementSpace();

    __shared__ Float p_in_block[in_block_size];
    __shared__ Float p_wei_block[wei_block_size];

    // register
    Float p_out_thread[out_hkwn_thread_desc.GetElementSpace()];

    // set threadwise output tensor to 0
    threadwise_4d_tensor_set_zero(out_hkwn_thread_desc, p_out_thread);

    for(unsigned c_block_data_begin = 0; c_block_data_begin < in_nchw_global_desc.GetLength(I1);
        c_block_data_begin += CPerBlock, __syncthreads())
    {
#if 1
        // input: global mem to LDS,
        //   convert [N,C,Hi,Wi] to [C,Hi,Wi,N]
        blockwise_4d_tensor_copy_reorder_by_get_dst_from_src<BlockSize>(
            in_nchw_global_desc,
            p_in_global + in_nchw_global_desc.Get1dIndex(n_block_data_begin,
                                                         c_block_data_begin,
                                                         hi_block_data_begin,
                                                         wi_block_data_begin),
            in_chwn_block_desc,
            p_in_block,
            in_nchw_block_desc.GetLengths(),
            reorder_chwn_from_nchw);
#else
        // input: global mem to LDS,
        //   no format conversion, this is wrong, for performance study only!
        blockwise_4d_tensor_copy<BlockSize>(in_nchw_global_desc,
                                            p_in_global +
                                                in_nchw_global_desc.Get1dIndex(n_block_data_begin,
                                                                               c_block_data_begin,
                                                                               hi_block_data_begin,
                                                                               wi_block_data_begin),
                                            in_nchw_block_desc,
                                            p_in_block,
                                            in_nchw_block_desc.GetLengths());
#endif

#if 1
        // weight: global mem to LDS,
        //   convert [K,C,S,R] to [S,R,C,K]
        blockwise_4d_tensor_copy_reorder_by_get_dst_from_src<BlockSize>(
            wei_kcsr_global_desc,
            p_wei_global +
                wei_kcsr_global_desc.Get1dIndex(k_block_data_begin, c_block_data_begin, 0, 0),
            wei_srck_block_desc,
            p_wei_block,
            wei_kcsr_block_desc.GetLengths(),
            reorder_srck_from_kcsr);
#else
        // weight: global mem to LDS,
        //   no format conversion, this is wrong, for performance study only!
        blockwise_4d_tensor_copy<BlockSize>(
            wei_kcsr_global_desc,
            p_wei_global +
                wei_kcsr_global_desc.Get1dIndex(k_block_data_begin, c_block_data_begin, 0, 0),
            wei_kcsr_block_desc,
            p_wei_block,
            wei_kcsr_block_desc.GetLengths());
#endif

        __syncthreads();

#if 1
        // a series of batched GEMM
        for(unsigned s = 0; s < S; ++s)
        {
            for(unsigned r = 0; r < R; ++r)
            {
                auto f_accum = [](auto& c, const auto&& ab) { c += ab; };

                blockwise_batch_gemm.run(p_wei_block + wei_srck_block_desc.Get1dIndex(s, r, 0, 0),
                                         p_in_block + in_chwn_block_desc.Get1dIndex(0, s, r, 0),
                                         p_out_thread,
                                         f_accum);
            }
        }
#endif
    }

    const auto matrix_c_index =
        blockwise_batch_gemm.CalculateThreadMatrixCIndex(get_thread_local_1d_id());

#if 0
    printf("%u %u, %u %u %u\n",get_block_1d_id(), get_thread_local_1d_id(), matrix_c_index.batch_begin, matrix_c_index.row_begin, matrix_c_index.col_begin);
#endif

    const unsigned ho_thread_data_begin = matrix_c_index.batch_begin;
    const unsigned k_thread_data_begin  = matrix_c_index.row_begin;
    const unsigned wo_thread_data_begin = matrix_c_index.col_begin / NPerThread;

#if 1
    // output: register to global mem,
    //   convert out_thread[Ho,K,Wo,N] to out_global[N,K,Ho,Wo]
    constexpr auto reorder_nkhw_from_hkwn = Sequence<3, 1, 0, 2>{};

    threadwise_4d_tensor_copy_reorder_by_get_dst_from_src(
        out_hkwn_thread_desc,
        p_out_thread,
        out_nkhw_global_desc,
        p_out_global + out_nkhw_global_desc.Get1dIndex(n_block_data_begin,
                                                       k_block_data_begin + k_thread_data_begin,
                                                       ho_block_data_begin + ho_thread_data_begin,
                                                       wo_block_data_begin + wo_thread_data_begin),
        out_hkwn_thread_desc.GetLengths(),
        reorder_nkhw_from_hkwn);
#else
    // output: register to global mem,
    //   no format conversion, assume register is in [N,K,Ho,Wo], this is wrong, for performance
    //   study only!
    constexpr auto out_nkhw_thread_desc =
        make_ConstantTensorDescriptor(Sequence<NPerThread, KPerThread, HoPerThread, WoPerThread>{});

    threadwise_4d_tensor_copy(
        out_nkhw_thread_desc,
        p_out_thread,
        out_nkhw_global_desc,
        p_out_global + out_nkhw_global_desc.Get1dIndex(n_block_data_begin,
                                                       k_block_data_begin + k_thread_data_begin,
                                                       ho_block_data_begin + ho_thread_data_begin,
                                                       wo_block_data_begin + wo_thread_data_begin),
        out_nkhw_thread_desc.GetLengths());
#endif
}
