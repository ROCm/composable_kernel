#pragma once
#include "common.cuh"
#include "ConstantTensorDescriptor.cuh"
#include "ConstantMatrixDescriptor.cuh"
#include "blockwise_4d_tensor_op.cuh"
#include "blockwise_2d_tensor_op.cuh"
#include "threadwise_2d_tensor_op.cuh"
#include "blockwise_gemm.cuh"

// define B = N*Hi*Wi
template <unsigned GridSize,
          unsigned BlockSize,
          class Float,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          unsigned BPerBlock,
          unsigned KPerBlock,
          unsigned CPerBlock,
          unsigned BPerThread,
          unsigned KPerThread,
          unsigned CPerThread,
          unsigned GemmThreadPerColumnPerCluster,
          unsigned GemmThreadPerRowPerCluster,
          unsigned InBlockCopyThreadPerDim0,
          unsigned InBlockCopyThreadPerDim1>
__global__ void gridwise_implicit_gemm_convolution_2_cnhw_srck_knhw_lds_pipeline(
    InGlobalDesc,
    Float* const __restrict__ p_in_global,
    WeiGlobalDesc,
    Float* const __restrict__ p_wei_global,
    OutGlobalDesc,
    Float* __restrict__ p_out_global)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_cnhw_global_desc  = InGlobalDesc{};
    constexpr auto wei_srck_global_desc = WeiGlobalDesc{};
    constexpr auto out_knhw_global_desc = OutGlobalDesc{};

    constexpr unsigned C  = in_cnhw_global_desc.GetLength(I0);
    constexpr unsigned N  = in_cnhw_global_desc.GetLength(I1);
    constexpr unsigned Hi = in_cnhw_global_desc.GetLength(I2);
    constexpr unsigned Wi = in_cnhw_global_desc.GetLength(I3);

    constexpr unsigned K  = out_knhw_global_desc.GetLength(I0);
    constexpr unsigned Ho = out_knhw_global_desc.GetLength(I2);
    constexpr unsigned Wo = out_knhw_global_desc.GetLength(I3);

    constexpr unsigned S = wei_srck_global_desc.GetLength(I0);
    constexpr unsigned R = wei_srck_global_desc.GetLength(I1);

    constexpr unsigned B          = N * Hi * Wi;
    constexpr unsigned BGhostRead = (S - 1) * Wi + (R - 1);

    // divide block work by 2d: [K, B]
    constexpr unsigned KBlockWork = (K + KPerBlock - 1) / KPerBlock;
    constexpr unsigned BBlockWork = (B + BPerBlock - 1) / BPerBlock;

    const unsigned k_block_work_id = get_block_1d_id() / BBlockWork;
    const unsigned b_block_work_id = get_block_1d_id() - k_block_work_id * BBlockWork;

    const unsigned k_block_data_begin = k_block_work_id * KPerBlock;
    const unsigned b_block_data_begin = b_block_work_id * BPerBlock;

#if 0
    if(get_thread_local_1d_id() == 0)
    {
        printf("K %u B %u, BGhostRead %u\n", K, B, BGhostRead);

        printf("%u %u, KBlockWork %u BBlockWork %u, k_block_data_begin %u b_block_data_begin %u\n",
               get_block_1d_id(),
               get_thread_local_1d_id(),
               KBlockWork,
               BBlockWork,
               k_block_data_begin,
               b_block_data_begin);
    }
#endif

    // flattend (2d) tensor view of gridwise input
    constexpr auto in_cb_global_desc = make_ConstantTensorDescriptor(Sequence<C, B>{});

    // tensor view of blockwise input and weight
    constexpr auto in_cb_block_desc =
        make_ConstantTensorDescriptor(Sequence<CPerBlock, BPerBlock + BGhostRead>{});

    constexpr auto wei_srck_block_desc =
        make_ConstantTensorDescriptor(Sequence<S, R, CPerBlock, KPerBlock>{});

    // tensor view of threadwise output in register
    constexpr auto out_kb_thread_desc =
        make_ConstantTensorDescriptor(Sequence<KPerThread, BPerThread>{});

#if 0
    if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(in_cb_block_desc, "in_cb_block_desc");
        print_ConstantTensorDescriptor(wei_srck_block_desc, "wei_srck_block_desc");
        print_ConstantTensorDescriptor(out_kb_thread_desc, "out_kb_thread_desc");

        printf("KPerBlock %u\n", KPerBlock);
    }
#endif

    // in: global mem to LDS
    //   formmat is [CPerBlock,BPerBlock + BGhostRead]
#if 1
    const auto blockwise_in_copy =
        Blockwise2dTensorCopy1<BlockSize,
                               Float,
                               decltype(in_cb_global_desc),
                               decltype(in_cb_block_desc),
                               decltype(in_cb_block_desc.GetLengths())>{};
#elif 1
    const auto blockwise_in_copy = Blockwise2dTensorCopy2<BlockSize,
                                                          Float,
                                                          decltype(in_cb_global_desc),
                                                          decltype(in_cb_block_desc),
                                                          decltype(in_cb_block_desc.GetLengths()),
                                                          InBlockCopyThreadPerDim0,
                                                          InBlockCopyThreadPerDim1>{};
#elif 0
    const auto blockwise_in_copy =
        blockwise_2d_tensor_copy_dummy_2<BlockSize,
                                         Float,
                                         decltype(in_cb_global_desc),
                                         decltype(in_cb_block_desc),
                                         decltype(in_cb_block_desc.GetLengths())>{};
#endif

    // weight: global mem to LDS,
    //   format is [S,R,CPerBlock,KPerBlock]
#if 1
    const auto blockwise_wei_copy =
        Blockwise4dTensorCopy1<BlockSize,
                               Float,
                               decltype(wei_srck_global_desc),
                               decltype(wei_srck_block_desc),
                               decltype(wei_srck_block_desc.GetLengths())>{};
#else
    const auto blockwise_wei_copy =
        blockwise_4d_tensor_copy_dummy<BlockSize,
                                       Float,
                                       decltype(wei_srck_global_desc),
                                       decltype(wei_srck_block_desc),
                                       decltype(wei_srck_block_desc.GetLengths())>{};
#endif

    // a series of blockwise GEMM
    // c_mtx += transpose(a_mtx) * b_mtx
    //   a_mtx and b_mtx saved in LDS, c_mtx saved in register
    //   a_mtx[C,K] is a sub-matrix of wei_block[S,R,C,K]
    //   b_mtx[C,B] is a subset of in_block[C,B + BGhostRead]
    //   c_mtx[K,B] is out_block[K,B]
    constexpr auto a_cxk_block_mtx_desc =
        make_ConstantMatrixDescriptor(Number<CPerBlock>{}, Number<KPerBlock>{});

    constexpr auto b_cxb_block_mtx_desc = make_ConstantMatrixDescriptor(
        Number<CPerBlock>{}, Number<BPerBlock>{}, Number<in_cb_block_desc.GetStride(I0)>{});

    constexpr auto c_kxb_thread_mtx_desc =
        make_ConstantMatrixDescriptor(Number<KPerThread>{}, Number<BPerThread>{});

    const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadC<BlockSize,
                                                                 decltype(a_cxk_block_mtx_desc),
                                                                 decltype(b_cxb_block_mtx_desc),
                                                                 decltype(c_kxb_thread_mtx_desc),
                                                                 true,
                                                                 false,
                                                                 false,
                                                                 CPerThread,
                                                                 GemmThreadPerColumnPerCluster,
                                                                 GemmThreadPerRowPerCluster,
                                                                 true>{};

    // LDS
    constexpr unsigned in_block_size  = in_cb_block_desc.GetElementSpace();
    constexpr unsigned wei_block_size = wei_srck_block_desc.GetElementSpace();

    // double buffer
    __shared__ Float p_in_block_0[in_block_size];
    __shared__ Float p_wei_block_0[wei_block_size];

    __shared__ Float p_in_block_1[in_block_size];
    __shared__ Float p_wei_block_1[wei_block_size];

    // register
    Float p_out_thread[out_kb_thread_desc.GetElementSpace()];

    // set threadwise output tensor to 0
    threadwise_2d_tensor_set_zero(out_kb_thread_desc, p_out_thread);

    // prelog: load data
#if 1
    // input: global mem to LDS,
    blockwise_in_copy.Run(p_in_global + in_cb_global_desc.Get1dIndex(0, b_block_data_begin),
                          p_in_block_0);
#endif

#if 1
    // weight: global mem to LDS,
    blockwise_wei_copy.Run(
        p_wei_global + wei_srck_global_desc.Get1dIndex(0, 0, 0, k_block_data_begin), p_wei_block_0);
#endif

    unsigned cloop = 0;

    for(unsigned c_block_data_begin = 0; c_block_data_begin + CPerBlock < C;
        c_block_data_begin += CPerBlock, ++cloop)
    {
        __syncthreads();

        Float* p_in_block_now  = (cloop % 2 == 0) ? p_in_block_0 : p_in_block_1;
        Float* p_wei_block_now = (cloop % 2 == 0) ? p_wei_block_0 : p_wei_block_1;

        Float* p_in_block_next  = (cloop % 2 == 0) ? p_in_block_1 : p_in_block_0;
        Float* p_wei_block_next = (cloop % 2 == 0) ? p_wei_block_1 : p_wei_block_0;

#if 1
        // preload next data
        // input: global mem to LDS,
        blockwise_in_copy.Run(p_in_global + in_cb_global_desc.Get1dIndex(
                                                c_block_data_begin + CPerBlock, b_block_data_begin),
                              p_in_block_next);
#endif

#if 1
        // weight: global mem to LDS,
        blockwise_wei_copy.Run(p_wei_global +
                                   wei_srck_global_desc.Get1dIndex(
                                       0, 0, c_block_data_begin + CPerBlock, k_block_data_begin),
                               p_wei_block_next);
#endif

        // a series of GEMM
        for(unsigned s = 0; s < S; ++s)
        {
            for(unsigned r = 0; r < R; ++r)
            {
                auto f_accum = [](auto& c, const auto&& ab) { c += ab; };

                blockwise_gemm.Run(p_wei_block_now + wei_srck_block_desc.Get1dIndex(s, r, 0, 0),
                                   p_in_block_now + s * Wi + r,
                                   p_out_thread,
                                   f_accum);
            }
        }
    }

    {
        // last cloop
        __syncthreads();

        Float* p_in_block_now  = (cloop % 2 == 0) ? p_in_block_0 : p_in_block_1;
        Float* p_wei_block_now = (cloop % 2 == 0) ? p_wei_block_0 : p_wei_block_1;

        // a series of GEMM
        for(unsigned s = 0; s < S; ++s)
        {
            for(unsigned r = 0; r < R; ++r)
            {
                auto f_accum = [](auto& c, const auto&& ab) { c += ab; };

                blockwise_gemm.Run(p_wei_block_now + wei_srck_block_desc.Get1dIndex(s, r, 0, 0),
                                   p_in_block_now + s * Wi + r,
                                   p_out_thread,
                                   f_accum);
            }
        }
    }

    // output: register to global mem,
    const auto matrix_c_index = blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

    const unsigned k_thread_data_begin = matrix_c_index.row;
    const unsigned b_thread_data_begin = matrix_c_index.col;

    const unsigned k_data_begin = k_block_data_begin + k_thread_data_begin;
    const unsigned b_data_begin = b_block_data_begin + b_thread_data_begin;

#if 0
    if(get_block_1d_id() == 0)
    {
        printf("%u %u, row %u col %u, k_data_begin %u b_data_begin %u, %f %f %f %f\n",
               get_block_1d_id(),
               get_thread_local_1d_id(),
               matrix_c_index.row,
               matrix_c_index.col,
               k_data_begin,
               b_data_begin,
               p_out_thread[0], p_out_thread[1], p_out_thread[2], p_out_thread[3]);
    }
#endif

    for(unsigned k = 0; k < out_kb_thread_desc.GetLength(I0); ++k)
    {
        for(unsigned b = 0; b < out_kb_thread_desc.GetLength(I1); ++b)
        {
            unsigned k_data = k_data_begin + k;
            unsigned b_data = b_data_begin + b;

            unsigned n_data = b_data / (Hi * Wi);
            unsigned itmp   = b_data - n_data * (Hi * Wi);
            unsigned h_data = itmp / Wi;
            unsigned w_data = itmp - h_data * Wi;

            if(n_data < N && h_data < Ho && w_data < Wo)
            {
                p_out_global[out_knhw_global_desc.Get1dIndex(k_data, n_data, h_data, w_data)] =
                    p_out_thread[out_kb_thread_desc.Get1dIndex(k, b)];
            }
        }
    }
}
