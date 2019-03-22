#pragma once
#include "common.hip.hpp"
#include "ConstantTensorDescriptor.hip.hpp"
#include "blockwise_2d_tensor_op.hip.hpp"
#include "blockwise_4d_tensor_op.hip.hpp"
#include "blockwise_direct_convolution.hip.hpp"
#include "threadwise_4d_tensor_op.hip.hpp"
#include "threadwise_direct_convolution.hip.hpp"

template <class TInWei,
          class TOut,
          class TAccum,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          unsigned ScalarPerVector,
          unsigned NPerBlock,
          unsigned KPerBlock,
          unsigned CPerBlock,
          unsigned HoPerBlock,
          unsigned WoPerBlock,
          unsigned NPerThread,
          unsigned KPerThread,
          unsigned CPerThread,
          unsigned HoPerThread,
          unsigned WoPerThread,
          unsigned InBlockCopyDataPerRead,
          unsigned WeiBlockCopyDataPerRead,
          unsigned BlockSize,
          unsigned GridSize>
__global__ void gridwise_direct_convolution_2_vectorized_nchw_kcyx_nkhw(
    const typename vector_type<TInWei,
                               ScalarPerVector>::MemoryType* const __restrict__ p_in_vec_global,
    const typename vector_type<TInWei,
                               ScalarPerVector>::MemoryType* const __restrict__ p_wei_vec_global,
    TOut* const __restrict__ p_out_global)
{
    using in_scalar_t     = TInWei;
    using in_vector_mem_t = typename vector_type<in_scalar_t, ScalarPerVector>::MemoryType;
    using out_scalar_t    = TOut;
    using accum_t         = TAccum;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_vec_global_desc  = InGlobalDesc{};
    constexpr auto wei_kcyx_vec_global_desc = WeiGlobalDesc{};
    constexpr auto out_nkhw_global_desc     = OutGlobalDesc{};

    constexpr unsigned N = in_nchw_vec_global_desc.GetLength(I0);
    constexpr unsigned K = wei_kcyx_vec_global_desc.GetLength(I0);
    constexpr unsigned C = wei_kcyx_vec_global_desc.GetLength(I1);
    constexpr unsigned Y = wei_kcyx_vec_global_desc.GetLength(I2);
    constexpr unsigned X = wei_kcyx_vec_global_desc.GetLength(I3);

    constexpr auto wei_ke_vec_global_desc = make_ConstantTensorDescriptor(
        Sequence<K, C * Y * X>{}); // 2d view of wei for blockwise copy

    constexpr unsigned HiPerBlock = HoPerBlock + Y - 1;
    constexpr unsigned WiPerBlock = WoPerBlock + X - 1;

    constexpr auto in_nchw_vec_block_desc = make_ConstantTensorDescriptor_aligned(
        Sequence<NPerBlock, CPerBlock, HiPerBlock, WiPerBlock>{}, Number<InBlockCopyDataPerRead>{});

    constexpr auto wei_ke_vec_block_desc = make_ConstantTensorDescriptor_aligned(
        Sequence<KPerBlock, CPerBlock * Y * X>{},
        Number<WeiBlockCopyDataPerRead>{}); // 2d view of wei for blockwise copy

    constexpr auto wei_kcyx_vec_block_desc =
        make_ConstantTensorDescriptor(Sequence<KPerBlock, CPerBlock, Y, X>{},
                                      Sequence<wei_ke_vec_block_desc.GetStride(I0), Y * X, X, 1>{});

    // shared mem
    constexpr unsigned in_block_size =
        in_nchw_vec_block_desc.GetElementSpace(Number<InBlockCopyDataPerRead>{});

    constexpr unsigned wei_block_size =
        wei_kcyx_vec_block_desc.GetElementSpace(Number<WeiBlockCopyDataPerRead>{});

    constexpr unsigned max_align = InBlockCopyDataPerRead > WeiBlockCopyDataPerRead
                                       ? InBlockCopyDataPerRead
                                       : WeiBlockCopyDataPerRead;

    __shared__ in_vector_mem_t
        p_in_vec_block[max_align * ((in_block_size + max_align - 1) / max_align)];
    __shared__ in_vector_mem_t
        p_wei_vec_block[max_align * ((wei_block_size + max_align - 1) / max_align)];

    // threadwise tensors
    constexpr unsigned HiPerThread = HoPerThread + Y - 1;
    constexpr unsigned WiPerThread = WoPerThread + X - 1;

    constexpr auto in_nchw_vec_thread_block_desc =
        make_ConstantTensorDescriptor(Sequence<NPerThread, CPerThread, HiPerThread, WiPerThread>{},
                                      in_nchw_vec_block_desc.GetStrides());

    constexpr auto wei_kcyx_vec_thread_block_desc = make_ConstantTensorDescriptor(
        Sequence<KPerThread, CPerThread, Y, X>{}, wei_kcyx_vec_block_desc.GetStrides());

    constexpr auto out_nkhw_thread_desc = get_convolution_output_default_4d_tensor_descriptor(
        in_nchw_vec_thread_block_desc, wei_kcyx_vec_thread_block_desc);

    // register
    out_scalar_t p_out_thread[out_nkhw_thread_desc.GetElementSpace()];

    // divide block work
    constexpr unsigned NBlockWork =
        (out_nkhw_global_desc.GetLength(I0) + NPerBlock - 1) / NPerBlock;
    constexpr unsigned KBlockWork =
        (out_nkhw_global_desc.GetLength(I1) + KPerBlock - 1) / KPerBlock;
    constexpr unsigned HBlockWork =
        (out_nkhw_global_desc.GetLength(I2) + HoPerBlock - 1) / HoPerBlock;
    constexpr unsigned WBlockWork =
        (out_nkhw_global_desc.GetLength(I3) + WoPerBlock - 1) / WoPerBlock;

    const unsigned block_id = blockIdx.x;

    unsigned itmp                  = block_id;
    const unsigned n_block_work_id = itmp / (KBlockWork * HBlockWork * WBlockWork);
    itmp -= n_block_work_id * (KBlockWork * HBlockWork * WBlockWork);
    const unsigned k_block_work_id = itmp / (HBlockWork * WBlockWork);
    itmp -= k_block_work_id * (HBlockWork * WBlockWork);
    const unsigned h_block_work_id = itmp / WBlockWork;
    const unsigned w_block_work_id = itmp - h_block_work_id * WBlockWork;

    const unsigned n_block_data_begin  = n_block_work_id * NPerBlock;
    const unsigned k_block_data_begin  = k_block_work_id * KPerBlock;
    const unsigned ho_block_data_begin = h_block_work_id * HoPerBlock;
    const unsigned wo_block_data_begin = w_block_work_id * WoPerBlock;

    const unsigned hi_block_data_begin = ho_block_data_begin; // minus padding
    const unsigned wi_block_data_begin = wo_block_data_begin; // minus padding

    // divide thread work
    constexpr unsigned NThreadWork = (NPerBlock + NPerThread - 1) / NPerThread;
    constexpr unsigned KThreadWork = (KPerBlock + KPerThread - 1) / KPerThread;
    constexpr unsigned HThreadWork = (HoPerBlock + HoPerThread - 1) / HoPerThread;
    constexpr unsigned WThreadWork = (WoPerBlock + WoPerThread - 1) / WoPerThread;

    const unsigned thread_id = threadIdx.x;

    itmp                            = thread_id;
    const unsigned n_thread_work_id = itmp / (KThreadWork * HThreadWork * WThreadWork);
    itmp -= n_thread_work_id * (KThreadWork * HThreadWork * WThreadWork);
    const unsigned k_thread_work_id = itmp / (HThreadWork * WThreadWork);
    itmp -= k_thread_work_id * (HThreadWork * WThreadWork);
    const unsigned h_thread_work_id = itmp / WThreadWork;
    const unsigned w_thread_work_id = itmp - h_thread_work_id * WThreadWork;

    const unsigned n_thread_data_begin  = n_thread_work_id * NPerThread;
    const unsigned k_thread_data_begin  = k_thread_work_id * KPerThread;
    const unsigned ho_thread_data_begin = h_thread_work_id * HoPerThread;
    const unsigned wo_thread_data_begin = w_thread_work_id * WoPerThread;

    const unsigned hi_thread_data_begin = ho_thread_data_begin;
    const unsigned wi_thread_data_begin = wo_thread_data_begin;

    constexpr auto blockwise_in_copy =
        Blockwise4dTensorCopy1<BlockSize,
                               in_vector_mem_t,
                               decltype(in_nchw_vec_global_desc),
                               decltype(in_nchw_vec_block_desc),
                               decltype(in_nchw_vec_block_desc.GetLengths()),
                               InBlockCopyDataPerRead>{};

#if 0
    constexpr auto blockwise_wei_copy =
        Blockwise4dTensorCopy1<BlockSize,
                               in_vector_mem_t,
                               decltype(wei_kcyx_vec_global_desc),
                               decltype(wei_kcyx_vec_block_desc),
                               decltype(wei_kcyx_vec_block_desc.GetLengths()),
                               1>{};
#elif 1
    const auto blockwise_wei_copy =
        Blockwise2dTensorCopy3<BlockSize,
                               in_vector_mem_t,
                               decltype(wei_ke_vec_global_desc),
                               decltype(wei_ke_vec_block_desc),
                               decltype(wei_ke_vec_block_desc.GetLengths()),
                               WeiBlockCopyDataPerRead>{};
#endif

#if 1 // debug
    // set threadwise output tensor to 0
    threadwise_4d_tensor_set_zero(out_nkhw_thread_desc, p_out_thread);
#endif

    for(unsigned c_block_data_begin = 0; c_block_data_begin < C;
        c_block_data_begin += CPerBlock, __syncthreads())
    {
        // copy input tensor to LDS
        blockwise_in_copy.Run(p_in_vec_global +
                                  in_nchw_vec_global_desc.Get1dIndex(n_block_data_begin,
                                                                     c_block_data_begin,
                                                                     hi_block_data_begin,
                                                                     wi_block_data_begin),
                              p_in_vec_block);

        // copy weight tensor to LDS
        blockwise_wei_copy.Run(
            p_wei_vec_global +
                wei_kcyx_vec_global_desc.Get1dIndex(k_block_data_begin, c_block_data_begin, 0, 0),
            p_wei_vec_block);

        __syncthreads();

        for(unsigned c_thread_data = 0; c_thread_data < CPerBlock; c_thread_data += CPerThread)
        {
// threadwise convolution
#if 1
            threadwise_direct_convolution_2(
                in_nchw_vec_thread_block_desc,
                p_in_vec_block +
                    in_nchw_vec_block_desc.Get1dIndex(n_thread_data_begin,
                                                      c_thread_data,
                                                      hi_thread_data_begin,
                                                      wi_thread_data_begin),
                wei_kcyx_vec_thread_block_desc,
                p_wei_vec_block +
                    wei_kcyx_vec_block_desc.Get1dIndex(k_thread_data_begin, c_thread_data, 0, 0),
                out_nkhw_thread_desc,
                p_out_thread);
#elif 0
            threadwise_direct_convolution_3(
                in_nchw_vec_thread_block_desc,
                p_in_vec_block +
                    in_nchw_vec_block_desc.Get1dIndex(n_thread_data_begin,
                                                      c_thread_data,
                                                      hi_thread_data_begin,
                                                      wi_thread_data_begin),
                wei_kcyx_vec_thread_block_desc,
                p_wei_vec_block +
                    wei_kcyx_vec_block_desc.Get1dIndex(k_thread_data_begin, c_thread_data, 0, 0),
                out_nkhw_thread_desc,
                p_out_thread);
#endif
        }
    }

    // copy output tensor from register to global mem
    threadwise_4d_tensor_copy(
        out_nkhw_thread_desc,
        p_out_thread,
        out_nkhw_global_desc,
        p_out_global +
            out_nkhw_global_desc.Get1dIndex(n_block_data_begin + n_thread_data_begin,
                                            k_block_data_begin + k_thread_data_begin,
                                            ho_block_data_begin + ho_thread_data_begin,
                                            wo_block_data_begin + wo_thread_data_begin),
        out_nkhw_thread_desc.GetLengths());
}
