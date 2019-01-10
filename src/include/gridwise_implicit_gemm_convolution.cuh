#pragma once
#include "constant_tensor_descriptor.cuh"
#include "blockwise_tensor_op.cuh"
#include "threadwise_tensor_op.cuh"

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
__global__ void gridwise_implicit_gemm_convolution_nchw_kcsr(InGlobalDesc,
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

    constexpr auto True  = Constant<bool, true>;
    constexpr auto False = Constant<bool, false>;

    constexpr auto in_nchw_global_desc  = InGlobalDesc{};
    constexpr auto wei_kcsr_global_desc = WeiGlobalDesc{};
    constexpr auto out_nkhw_global_desc = OutGlobalDesc{};

    constexpr unsigned S = wei_kcsr_global_desc.GetLength(I2);
    constexpr unsigned R = wei_kcsr_global_desc.GetLength(I3);

    constexpr unsigned HiPerBlock = HoPerBlock + S - 1;
    constexpr unsigned WiPerBlock = WoPerBlock + R - 1;

    // block
    constexpr auto in_chwn_block_desc =
        make_ConstantTensorDescriptor(Sequence<CPerBlock, HiPerBlock, WiPerBlock, NPerBlock>{});

    constexpr auto wei_srck_block_desc =
        make_ConstantTensorDescriptor(Sequence<S, R, CPerBlock, KPerBlock>{});

    // LDS
    constexpr unsigned in_block_size  = in_chwn_block_desc.GetElementSpace();
    constexpr unsigned wei_block_size = wei_srck_block_desc.GetElementSpace();

    __shared__ Float p_in_block[in_block_size];
    __shared__ Float p_wei_block[wei_block_size];

    // thread
    constexpr auto out_hkwn_thread_desc = xxxxxx();

    // register
    Float p_out_thread[out_hkwn_thread_desc.GetElementSpace()];

    // set threadwise output tensor to 0
    threadwise_4d_tensor_set_zero(out_hkwn_thread_desc, p_out_thread);

    for(unsigned c_block_data_begin = 0; c_block_data_begin < in_global_desc.GetLength(I1);
        c_block_data_begin += CPerBlock, __syncthreads())
    {
        // input: global mem to LDS,
        //   convert 4d-tensor in[N,C,Hi,Wi] to matrix in_matrix[C,Hi*Wi*N]
        constexpr auto reorder_nchw2chwn = Sequence<3, 0, 1, 2>{};

        blockwise_4d_tensor_copy_reorder<BlockSize>(in_nchw_global_desc,
                                                    p_in_global,
                                                    in_chwn_block_desc,
                                                    p_in_block,
                                                    in_chwn_block_desc,
                                                    reorder_nchw2chwn);

        // matrix view of input
        constexpr unsigned in_row = in_chwn_block_desc.GetLength(I0);
        constexpr unsigned in_col = in_chwn_block_desc.GetLength(I1) *
                                    in_chwn_block_desc.GetLength(I2) *
                                    in_chwn_block_desc.GetLength(I3);
        constexpr auto in_cxhwn_block_mtx_desc =
            make_ConstantMatrixDescriptor(Number<in_row>, Number<in_col>, Number<in_col>);

        // weight: global mem to LDS,
        //   convert 4d-tensor wei[K,C,S,R] to matrix wei_matrix[S*R*C,K]
        constexpr auto reorder_kcsr2srck = Sequence<3, 2, 0, 1>{};

        blockwise_4d_tensor_copy_reorder<BlockSize>(wei_csrk_global_desc,
                                                    p_wei_global,
                                                    wei_csrk_block_desc,
                                                    p_wei_block,
                                                    wei_csrk_block_desc,
                                                    reorder_kcsr2csrk);

        // matrix view of wei
        constexpr unsigned wei_row = wei_srck_block_desc.GetLength(I0) *
                                     wei_srck_block_desc.GetLength(I1) *
                                     wei_srck_block_desc.GetLength(I2);
        constexpr unsigned wei_col = wei_srck_block_desc.GetLength(I3);
        constexpr auto wei_srcxk_block_mtx_desc =
            make_ConstantMatrixDescriptor(Number<wei_row>, Number<wei_col>, Number<wei_col>);

        __syncthreads();

        // a series of batched GEMM
        // blockwise batched GEMM, C_matrix += transpose(A_matrix) * B_matrix
        //   A_matrix and B_matrix saved in LDS, c_matrix saved in register
        //   A_matrix[C,K] is a sub-matrix of wei_matrix[S*R*C,K]
        //   B_matrix[C,Wo*N] is a sub-matrix of in_matrix[C,Hi*Wi*N]
        //   C_matrix[K,Wo*N] is a sub-matrix of out_matrix[Ho*K,Wo*N]
        constexpr auto a_block_mtx_desc = wei_srcxk_block_mtx_desc.MakeSubMatrixDescriptor(
            Number<CPerBlock>{}, Number<KPerBlock>{});

        constexpr auto b_block_mtx_desc = in_cxhwn_block_mtx_desc.MakeSubMatrixDescriptor(
            Number<CPerBlock>{}, Number<WoPerBlock * NPerBlock>{});

        auto f_accum = (auto& c, auto& v) { c += v; };

        const auto blockwise_batch_gemm =
            blockwise_1d_strided_batched_gemm_block_a_block_b_thread_c<BlockSize,
                                                                       a_block_mtx_desc,
                                                                       b_block_mtx_desc,
                                                                       true,
                                                                       false,
                                                                       HoPerBlock,
                                                                       0,
                                                                       xxx_b_matrix_stride,
                                                                       HoPerThread,
                                                                       KPerThread,
                                                                       NPerThread * WoPerThread,
                                                                       CPerTrhead,
                                                                       decltype(f_accum)>{};
        // loop over filter point
        for(unsigned s = 0; s < S; ++s)
        {
            for(unsigned r = 0; r < R; ++r)
            {
                blockwise_batch_gemm.run(
                    p_wei_block + wei_srcxk_block_mtx_desc.Get1dIndex(xxxxx, xxxx),
                    p_in_block + in_cxhwn_block_mtx_desc.Get1dIndex(xxxx, xxxx),
                    p_out_thread);
            }
        }
    }

    const auto matrix_c_index =
        blockwise_batch_gemm.CalculateThreadMatrixCIndex(get_thread_local_id());

    const unsigned ho_thread_data_begin = matrix_c_index.batch_begin;
    const unsigned k_thread_data_begin  = matrix_c_index.col_begin;
    const unsigned wo_thread_data_begin = matrix_c_index.row_begin / NPerThread;

    // output: register to global mem,
    //   convert matrix out_matrix[Ho*K,Wo*N] to 4d-tensor out[N,K,Ho,Wo]
    constexpr auto reorder_hkwn2nkhw = Sequence<2, 1, 3, 0>{};
    threadwise_4d_tensor_copy_reorder(
        out_hkwn_thread_desc,
        p_out_thread,
        out_nkhw_global_desc,
        p_out_global + out_nkhw_global_desc.GetIndex(n_block_data_begin,
                                                     k_block_data_begin + k_thread_data_begin,
                                                     ho_block_data_begin + ho_thread_data_begin,
                                                     wo_block_data_begin + wo_thread_data_begin),
        out_hkwn_thread_desc,
        reorder_hkwn2nkhw);
}
