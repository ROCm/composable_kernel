#pragma once
#include "constant_tensor_descriptor.cuh"
#include "threadwise_tensor_op.cuh"
#include "threadwise_convolution.cuh"

template <class TFloat,
          class InBlockDesc,
          class WeiBlockDesc,
          class OutBlockDesc,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned BlockSize>
__device__ void blockwise_convolution(InBlockDesc,
                                      TFloat* const __restrict__ p_in_block,
                                      WeiBlockDesc,
                                      TFloat* const __restrict__ p_wei_block,
                                      OutBlockDesc,
                                      TFloat* __restrict__ p_out_block)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto in_block_desc  = InBlockDesc{};
    constexpr auto wei_block_desc = WeiBlockDesc{};
    constexpr auto out_block_desc = OutBlockDesc{};

    constexpr unsigned S = wei_block_desc.GetLength(I2);
    constexpr unsigned R = wei_block_desc.GetLength(I3);

    constexpr unsigned NPerBlock = out_block_desc.GetLength(I0);
    constexpr unsigned KPerBlock = out_block_desc.GetLength(I1);
    constexpr unsigned YPerBlock = (out_block_desc.GetLength(I2) + OutTileSizeH - 1) / OutTileSizeH;
    constexpr unsigned XPerBlock = (out_block_desc.GetLength(I3) + OutTileSizeW - 1) / OutTileSizeW;

    constexpr unsigned CPerBlock = in_block_desc.GetLength(I1);

    constexpr unsigned InTileSizeH = OutTileSizeH + S - 1;
    constexpr unsigned InTileSizeW = OutTileSizeW + R - 1;

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(in_block_desc);
        print_ConstantTensorDescriptor(wei_block_desc);
        print_ConstantTensorDescriptor(out_block_desc);
    }
#endif

    constexpr auto in_thread_src_desc = make_ConstantTensorDescriptor(
        Sequence<1, CPerBlock, InTileSizeH, InTileSizeW>{}, in_block_desc.GetStrides());

    constexpr auto wei_thread_src_desc =
        make_ConstantTensorDescriptor(Sequence<1, CPerBlock, S, R>{}, wei_block_desc.GetStrides());

    constexpr auto out_thread_src_desc = make_ConstantTensorDescriptor(
        Sequence<1, 1, OutTileSizeH, OutTileSizeW>{}, out_block_desc.GetStrides());

    constexpr auto in_thread_dst_desc =
        make_ConstantTensorDescriptor(in_thread_src_desc.GetLengths());

    constexpr auto wei_thread_dst_desc =
        make_ConstantTensorDescriptor(wei_thread_src_desc.GetLengths());

    constexpr auto out_thread_dst_desc =
        make_ConstantTensorDescriptor(out_thread_src_desc.GetLengths());

    const unsigned thread_id = threadIdx.x;

    for(unsigned thread_work_id = thread_id; thread_work_id < NPerBlock * YPerBlock * XPerBlock;
        thread_work_id += BlockSize)
    {
        unsigned itmp             = thread_work_id;
        unsigned n_thread_work_id = itmp / (YPerBlock * XPerBlock);
        itmp -= n_thread_work_id * (YPerBlock * XPerBlock);
        unsigned y_thread_work_id = itmp / XPerBlock;
        unsigned x_thread_work_id = itmp - y_thread_work_id * XPerBlock;

        unsigned n_thread_work_begin  = n_thread_work_id * 1;
        unsigned ho_thread_work_begin = y_thread_work_id * OutTileSizeH;
        unsigned wo_thread_work_begin = x_thread_work_id * OutTileSizeW;

        unsigned hi_thread_work_begin = ho_thread_work_begin; // minus padding
        unsigned wi_thread_work_begin = wo_thread_work_begin; // minus padding

        TFloat p_in_thread[in_thread_src_desc.GetElementSpace()];
        TFloat p_wei_thread[wei_thread_src_desc.GetElementSpace()];
        TFloat p_out_thread[out_thread_src_desc.GetElementSpace()];

        auto f_copy = [](const TFloat& src, TFloat& dst) { dst = src; };

        // copy input tensor into register
        threadwise_4d_tensor_op_binary<TFloat,
                                       decltype(in_thread_src_desc),
                                       decltype(in_thread_dst_desc),
                                       decltype(f_copy)>(
            in_thread_src_desc,
            p_in_block + in_block_desc.Get1dIndex(
                             n_thread_work_begin, 0, hi_thread_work_begin, wi_thread_work_begin),
            in_thread_dst_desc,
            p_in_thread,
            f_copy);

        for(unsigned k_thread_work_begin = 0; k_thread_work_begin < KPerBlock;
            ++k_thread_work_begin)
        {
            // copy weight tensor into register
            threadwise_4d_tensor_op_binary<TFloat,
                                           decltype(wei_thread_src_desc),
                                           decltype(wei_thread_dst_desc),
                                           decltype(f_copy)>(
                wei_thread_src_desc,
                p_wei_block + wei_block_desc.Get1dIndex(k_thread_work_begin, 0, 0, 0),
                wei_thread_dst_desc,
                p_wei_thread,
                f_copy);

            // copy output tensor into register
            threadwise_4d_tensor_op_binary<TFloat,
                                           decltype(out_thread_src_desc),
                                           decltype(out_thread_dst_desc),
                                           decltype(f_copy)>(
                out_thread_src_desc,
                p_out_block + out_block_desc.Get1dIndex(n_thread_work_begin,
                                                        k_thread_work_begin,
                                                        ho_thread_work_begin,
                                                        wo_thread_work_begin),
                out_thread_dst_desc,
                p_out_thread,
                f_copy);

            // threadwise convolution
            threadwise_direct_convolution<TFloat,
                                          decltype(in_thread_dst_desc),
                                          decltype(wei_thread_dst_desc),
                                          decltype(out_thread_dst_desc)>(in_thread_dst_desc,
                                                                         p_in_thread,
                                                                         wei_thread_dst_desc,
                                                                         p_wei_thread,
                                                                         out_thread_dst_desc,
                                                                         p_out_thread);

            // accumulate output tensor into LDS
            threadwise_4d_tensor_op_binary<TFloat,
                                           decltype(out_thread_dst_desc),
                                           decltype(out_thread_src_desc),
                                           decltype(f_copy)>(
                out_thread_dst_desc,
                p_out_thread,
                out_thread_src_desc,
                p_out_block + out_block_desc.Get1dIndex(n_thread_work_begin,
                                                        k_thread_work_begin,
                                                        ho_thread_work_begin,
                                                        wo_thread_work_begin),
                f_copy);
        }
    }
}
