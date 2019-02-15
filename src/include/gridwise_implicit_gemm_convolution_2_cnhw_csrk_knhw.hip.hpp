#pragma once
#include "common.hip.hpp"
#include "ConstantTensorDescriptor.hip.hpp"
#include "ConstantMatrixDescriptor.hip.hpp"
#include "blockwise_4d_tensor_op.hip.hpp"
#include "blockwise_2d_tensor_op.hip.hpp"
#include "threadwise_2d_tensor_op.hip.hpp"
#include "blockwise_gemm.hip.hpp"

// define B = flatten(N, Hi, Wi)
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
          unsigned GemmThreadPerColumnPerCluster,
          unsigned GemmThreadPerRowPerCluster,
          unsigned GemmMPerThreadSubC,
          unsigned GemmNPerThreadSubC,
          unsigned GemmMLevel0Cluster,
          unsigned GemmNLevel0Cluster,
          unsigned GemmMLevel1Cluster,
          unsigned GemmNLevel1Cluster,
          unsigned GemmKPerThreadLoop,
          unsigned InBlockCopyThreadPerDim0,
          unsigned InBlockCopyThreadPerDim1,
          unsigned WeiBlockCopyThreadPerDim0,
          unsigned WeiBlockCopyThreadPerDim1,
          unsigned InBlockCopyDataPerRead,
          unsigned WeiBlockCopyDataPerRead>
__global__ void
gridwise_implicit_gemm_convolution_2_cnhw_csrk_knhw(const Float* const __restrict__ p_in_global,
                                                    const Float* const __restrict__ p_wei_global,
                                                    Float* const __restrict__ p_out_global)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_cnhw_global_desc  = InGlobalDesc{};
    constexpr auto wei_csrk_global_desc = WeiGlobalDesc{};
    constexpr auto out_knhw_global_desc = OutGlobalDesc{};

    constexpr unsigned C  = in_cnhw_global_desc.GetLength(I0);
    constexpr unsigned N  = in_cnhw_global_desc.GetLength(I1);
    constexpr unsigned Hi = in_cnhw_global_desc.GetLength(I2);
    constexpr unsigned Wi = in_cnhw_global_desc.GetLength(I3);

    constexpr unsigned K  = out_knhw_global_desc.GetLength(I0);
    constexpr unsigned Ho = out_knhw_global_desc.GetLength(I2);
    constexpr unsigned Wo = out_knhw_global_desc.GetLength(I3);

    constexpr unsigned S = wei_csrk_global_desc.GetLength(I1);
    constexpr unsigned R = wei_csrk_global_desc.GetLength(I2);

    constexpr unsigned B          = N * Hi * Wi;
    constexpr unsigned BGhostRead = (S - 1) * Wi + (R - 1);

    // divide block work by 2d: [K, B]
    constexpr unsigned KBlockWork = (K + KPerBlock - 1) / KPerBlock;
    constexpr unsigned BBlockWork = (B + BPerBlock - 1) / BPerBlock;

    const unsigned k_block_work_id = get_block_1d_id() / BBlockWork;
    const unsigned b_block_work_id = get_block_1d_id() - k_block_work_id * BBlockWork;

    const unsigned k_block_data_begin = k_block_work_id * KPerBlock;
    const unsigned b_block_data_begin = b_block_work_id * BPerBlock;

    // flattend (2d) tensor view of gridwise input
    constexpr auto in_cb_global_desc  = make_ConstantTensorDescriptor(Sequence<C, B>{});
    constexpr auto wei_ek_global_desc = make_ConstantTensorDescriptor(Sequence<C * S * R, K>{});

    // tensor view of blockwise input and weight
    //   be careful of alignment
    constexpr auto in_cb_block_desc = make_ConstantTensorDescriptor_aligned(
        Sequence<CPerBlock, BPerBlock + BGhostRead>{}, Number<InBlockCopyDataPerRead>{});

    constexpr auto wei_ek_block_desc = make_ConstantTensorDescriptor_aligned(
        Sequence<CPerBlock * S * R, KPerBlock>{}, Number<WeiBlockCopyDataPerRead>{});

    constexpr auto wei_csrk_block_desc = make_ConstantTensorDescriptor_aligned(
        Sequence<CPerBlock, S, R, KPerBlock>{}, Number<WeiBlockCopyDataPerRead>{});

    // tensor view of threadwise output in register
    constexpr auto out_kb_thread_desc =
        make_ConstantTensorDescriptor(Sequence<KPerThread, BPerThread>{});

#if 0
    if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(in_cnhw_global_desc, "in_cnhw_global_desc");
        print_ConstantTensorDescriptor(wei_csrk_global_desc, "wei_csrk_global_desc");
        print_ConstantTensorDescriptor(out_knhw_global_desc, "out_knhw_global_desc");

        print_ConstantTensorDescriptor(in_cb_global_desc, "in_cb_global_desc");
        print_ConstantTensorDescriptor(wei_ek_global_desc, "wei_ek_global_desc");

        print_ConstantTensorDescriptor(in_cb_block_desc, "in_cb_block_desc");
        print_ConstantTensorDescriptor(wei_csrk_block_desc, "wei_csrk_block_desc");
        print_ConstantTensorDescriptor(wei_ek_block_desc, "wei_ek_block_desc");
        print_ConstantTensorDescriptor(out_kb_thread_desc, "out_kb_thread_desc");

        printf("KPerBlock %u\n", KPerBlock);
    }
#endif

// blockwise in copy
//   formmat is [CPerBlock,BPerBlock + BGhostRead]
#if 0
    const auto blockwise_in_copy =
        Blockwise2dTensorCopy1<BlockSize,
                               Float,
                               decltype(in_cb_global_desc),
                               decltype(in_cb_block_desc),
                               decltype(in_cb_block_desc.GetLengths())>{};
#elif 0
    const auto blockwise_in_copy  = Blockwise2dTensorCopy2<BlockSize,
                                                          Float,
                                                          decltype(in_cb_global_desc),
                                                          decltype(in_cb_block_desc),
                                                          decltype(in_cb_block_desc.GetLengths()),
                                                          InBlockCopyThreadPerDim0,
                                                          InBlockCopyThreadPerDim1>{};
#elif 1
    const auto blockwise_in_copy  = Blockwise2dTensorCopy3<BlockSize,
                                                          Float,
                                                          decltype(in_cb_global_desc),
                                                          decltype(in_cb_block_desc),
                                                          decltype(in_cb_block_desc.GetLengths()),
                                                          InBlockCopyDataPerRead>{};
#endif

// blockwise wei copy
//   format is [CPerBlock*S*R,KPerBlock]
#if 0
    const auto blockwise_wei_copy =
        Blockwise2dTensorCopy1<BlockSize,
                               Float,
                               decltype(wei_ek_global_desc),
                               decltype(wei_ek_block_desc),
                               decltype(wei_ek_block_desc.GetLengths())>{};
#elif 0
    const auto blockwise_wei_copy = Blockwise2dTensorCopy2<BlockSize,
                                                           Float,
                                                           decltype(wei_ek_global_desc),
                                                           decltype(wei_ek_block_desc),
                                                           decltype(wei_ek_block_desc.GetLengths()),
                                                           WeiBlockCopyThreadPerDim0,
                                                           WeiBlockCopyThreadPerDim1>{};
#elif 1
    const auto blockwise_wei_copy = Blockwise2dTensorCopy3<BlockSize,
                                                           Float,
                                                           decltype(wei_ek_global_desc),
                                                           decltype(wei_ek_block_desc),
                                                           decltype(wei_ek_block_desc.GetLengths()),
                                                           WeiBlockCopyDataPerRead>{};
#endif

    // a series of blockwise GEMM
    // c_mtx += transpose(a_mtx) * b_mtx
    //   a_mtx and b_mtx saved in LDS, c_mtx saved in register
    //   a_mtx[C,K] is a sub-matrix of wei_block[C,S,R,K]
    //   b_mtx[C,B] is a subset of in_block[C,B + BGhostRead]
    //   c_mtx[K,B] is out_block[K,B]
    constexpr auto a_cxk_block_mtx_desc = make_ConstantMatrixDescriptor(
        Number<CPerBlock>{}, Number<KPerBlock>{}, Number<wei_csrk_block_desc.GetStride(I0)>{});

    constexpr auto b_cxb_block_mtx_desc = make_ConstantMatrixDescriptor(
        Number<CPerBlock>{}, Number<BPerBlock>{}, Number<in_cb_block_desc.GetStride(I0)>{});

    constexpr auto c_kxb_thread_mtx_desc =
        make_ConstantMatrixDescriptor(Number<KPerThread>{}, Number<BPerThread>{});

#if 0
    const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadC<BlockSize,
                                                                 decltype(a_cxk_block_mtx_desc),
                                                                 decltype(b_cxb_block_mtx_desc),
                                                                 decltype(c_kxb_thread_mtx_desc),
                                                                 true,
                                                                 false,
                                                                 false,
                                                                 GemmKPerThreadLoop,
                                                                 GemmThreadPerColumnPerCluster,
                                                                 GemmThreadPerRowPerCluster,
                                                                 true>{};
#else
    const auto blockwise_gemm =
        BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<BlockSize,
                                                                decltype(a_cxk_block_mtx_desc),
                                                                decltype(b_cxb_block_mtx_desc),
                                                                decltype(c_kxb_thread_mtx_desc),
                                                                GemmMPerThreadSubC,
                                                                GemmNPerThreadSubC,
                                                                GemmMLevel0Cluster,
                                                                GemmNLevel0Cluster,
                                                                GemmMLevel1Cluster,
                                                                GemmNLevel1Cluster,
                                                                GemmKPerThreadLoop>{};
#endif

    // LDS: be careful of alignment
    constexpr unsigned in_block_size =
        in_cb_block_desc.GetElementSpace(Number<InBlockCopyDataPerRead>{});

    constexpr unsigned wei_block_size =
        wei_csrk_block_desc.GetElementSpace(Number<WeiBlockCopyDataPerRead>{});

    constexpr unsigned max_align = InBlockCopyDataPerRead > WeiBlockCopyDataPerRead
                                       ? InBlockCopyDataPerRead
                                       : WeiBlockCopyDataPerRead;

    __shared__ Float p_in_block[max_align * ((in_block_size + max_align - 1) / max_align)];
    __shared__ Float p_wei_block[max_align * ((wei_block_size + max_align - 1) / max_align)];

    // register
    Float p_out_thread[out_kb_thread_desc.GetElementSpace()];

    // set threadwise output tensor to 0
    threadwise_2d_tensor_set_zero(out_kb_thread_desc, p_out_thread);

    const Float* p_in_global_block_offset =
        p_in_global + in_cb_global_desc.Get1dIndex(0, b_block_data_begin);

    const Float* p_wei_global_block_offset =
        p_wei_global + wei_csrk_global_desc.Get1dIndex(0, 0, 0, k_block_data_begin);

    for(unsigned c_block_data_begin = 0; c_block_data_begin < C; c_block_data_begin += CPerBlock,
                 p_in_global_block_offset += CPerBlock * in_cb_global_desc.GetStride(I0),
                 p_wei_global_block_offset += CPerBlock * wei_csrk_global_desc.GetStride(I0),
                 __syncthreads())
    {
        // input: global mem to LDS,
        blockwise_in_copy.Run(p_in_global_block_offset, p_in_block);

        // weight: global mem to LDS,
        blockwise_wei_copy.Run(p_wei_global_block_offset, p_wei_block);

        __syncthreads();

        // a series of GEMM
        for(unsigned s = 0; s < S; ++s)
        {
            for(unsigned r = 0; r < R; ++r)
            {
                auto f_accum = [](auto& acc, const auto&& v) { acc += v; };

#if 1
                blockwise_gemm.Run
#elif 0
                blockwise_gemm.Run_v2
#elif 0
                blockwise_gemm.Run_RegisterDoubleBuffer
#endif
                    (p_wei_block + wei_csrk_block_desc.Get1dIndex(0, s, r, 0),
                     p_in_block + s * Wi + r,
                     p_out_thread,
                     f_accum);
            }
        }
    }

    // output: register to global mem,
    const auto c_thread_mtx_begin =
        blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

    const unsigned k_thread_data_begin = k_block_data_begin + c_thread_mtx_begin.row;
    const unsigned b_thread_data_begin = b_block_data_begin + c_thread_mtx_begin.col;

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
            const auto c_thread_mtx_distance =
                blockwise_gemm.GetDistanceFromBeginOfThreadMatrixC(k, b);

            unsigned k_data = k_thread_data_begin + c_thread_mtx_distance.row;
            unsigned b_data = b_thread_data_begin + c_thread_mtx_distance.col;

            unsigned n_data = b_data / (Hi * Wi);
            unsigned itmp   = b_data - n_data * (Hi * Wi);
            unsigned h_data = itmp / Wi;
            unsigned w_data = itmp - h_data * Wi;

#if 0
            if(get_block_1d_id() == 0)
            {
                printf("%u %u, k %u b %u, k_data %u n_data %u h_data %u w_data %u %f\n",
                       get_block_1d_id(),
                       get_thread_local_1d_id(),
                       k,
                       b,
                       k_data,
                       n_data,
                       h_data,
                       w_data,
                       p_out_thread[out_kb_thread_desc.Get1dIndex(k, b)]);
            }
#endif
            if(n_data < N && h_data < Ho && w_data < Wo)
            {
                p_out_global[out_knhw_global_desc.Get1dIndex(k_data, n_data, h_data, w_data)] =
                    p_out_thread[out_kb_thread_desc.Get1dIndex(k, b)];
            }
        }
    }
}
