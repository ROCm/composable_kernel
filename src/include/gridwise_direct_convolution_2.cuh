#pragma once
#include "constant_tensor_descriptor.cuh"
#include "blockwise_tensor_op.cuh"
#include "blockwise_direct_convolution.cuh"
#include "threadwise_tensor_op.cuh"
#include "threadwise_direct_convolution.cuh"

template <class TFloat,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned NPerBlock,
          unsigned KPerBlock,
          unsigned CPerBlock,
          unsigned YPerBlock,
          unsigned XPerBlock,
          unsigned NPerThread,
          unsigned KPerThread,
          unsigned CPerThread,
          unsigned NBlockOpLen0,
          unsigned NBlockOpLen1,
          unsigned NBlockOpLen2,
          unsigned NBlockOpLen3,
          unsigned BlockSize,
          unsigned GridSize>
__global__ void gridwise_direct_convolution_2(InGlobalDesc,
                                              TFloat* const __restrict__ p_in_global,
                                              WeiGlobalDesc,
                                              TFloat* const __restrict__ p_wei_global,
                                              OutGlobalDesc,
                                              TFloat* __restrict__ p_out_global)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto in_global_desc  = InGlobalDesc{};
    constexpr auto wei_global_desc = WeiGlobalDesc{};
    constexpr auto out_global_desc = OutGlobalDesc{};

    constexpr unsigned S = wei_global_desc.GetLength(I2);
    constexpr unsigned R = wei_global_desc.GetLength(I3);

    constexpr unsigned HoPerBlock = OutTileSizeH * YPerBlock;
    constexpr unsigned WoPerBlock = OutTileSizeW * XPerBlock;

    constexpr unsigned HiPerBlock = YPerBlock * OutTileSizeH + S - 1;
    constexpr unsigned WiPerBlock = XPerBlock * OutTileSizeW + R - 1;

    constexpr auto in_block_global_desc = make_ConstantTensorDescriptor(
        Sequence<NPerBlock, CPerBlock, HiPerBlock, WiPerBlock>{}, in_global_desc.GetStrides());

    constexpr auto wei_block_global_desc = make_ConstantTensorDescriptor(
        Sequence<KPerBlock, CPerBlock, S, R>{}, wei_global_desc.GetStrides());

    constexpr auto in_block_desc = make_ConstantTensorDescriptor(in_block_global_desc.GetLengths());
    constexpr auto wei_block_desc =
        make_ConstantTensorDescriptor(wei_block_global_desc.GetLengths());

    // shared mem
    constexpr unsigned in_block_size  = in_block_desc.GetElementSpace();
    constexpr unsigned wei_block_size = wei_block_desc.GetElementSpace();

    __shared__ TFloat p_in_block[in_block_size];
    __shared__ TFloat p_wei_block[wei_block_size];

    // threadwise tensors
    constexpr unsigned InTileSizeH = OutTileSizeH + S - 1;
    constexpr unsigned InTileSizeW = OutTileSizeW + R - 1;

    constexpr auto in_thread_block_desc = make_ConstantTensorDescriptor(
        Sequence<NPerThread, CPerThread, InTileSizeH, InTileSizeW>{}, in_block_desc.GetStrides());

    constexpr auto wei_thread_block_desc = make_ConstantTensorDescriptor(
        Sequence<KPerThread, CPerThread, S, R>{}, wei_block_desc.GetStrides());

    constexpr auto in_thread_desc =
        make_ConstantTensorDescriptor(in_thread_block_desc.GetLengths());
    constexpr auto wei_thread_desc =
        make_ConstantTensorDescriptor(wei_thread_block_desc.GetLengths());
    constexpr auto out_thread_desc =
        get_output_4d_tensor_descriptor(in_thread_desc, wei_thread_desc);

    constexpr auto out_thread_global_desc =
        make_ConstantTensorDescriptor(out_thread_desc.GetLengths(), out_global_desc.GetStrides());

    // register
    constexpr unsigned in_thread_size  = in_thread_desc.GetElementSpace();
    constexpr unsigned wei_thread_size = wei_thread_desc.GetElementSpace();
    constexpr unsigned out_thread_size = out_thread_desc.GetElementSpace();

    TFloat p_in_thread[in_thread_size];
    TFloat p_wei_thread[wei_thread_size];
    TFloat p_out_thread[out_thread_size];

    // divide block work
    constexpr unsigned NBlockWork = (out_global_desc.GetLength(I0) + NPerBlock - 1) / NPerBlock;
    constexpr unsigned KBlockWork = (out_global_desc.GetLength(I1) + KPerBlock - 1) / KPerBlock;
    constexpr unsigned YBlockWork = (out_global_desc.GetLength(I2) + HoPerBlock - 1) / HoPerBlock;
    constexpr unsigned XBlockWork = (out_global_desc.GetLength(I3) + WoPerBlock - 1) / WoPerBlock;

    const unsigned block_id = blockIdx.x;

    unsigned itmp                  = block_id;
    const unsigned n_block_work_id = itmp / (KBlockWork * YBlockWork * XBlockWork);
    itmp -= n_block_work_id * (KBlockWork * YBlockWork * XBlockWork);
    const unsigned k_block_work_id = itmp / (YBlockWork * XBlockWork);
    itmp -= k_block_work_id * (YBlockWork * XBlockWork);
    const unsigned y_block_work_id = itmp / XBlockWork;
    const unsigned x_block_work_id = itmp - y_block_work_id * XBlockWork;

    const unsigned n_block_data_offset = n_block_work_id * NPerBlock;
    const unsigned k_block_data_offset = k_block_work_id * KPerBlock;
    const unsigned y_block_data_offset = y_block_work_id * YPerBlock;
    const unsigned x_block_data_offset = x_block_work_id * XPerBlock;

    const unsigned ho_block_data_offset = y_block_data_offset * OutTileSizeH;
    const unsigned wo_block_data_offset = x_block_data_offset * OutTileSizeW;

    const unsigned hi_block_data_offset = ho_block_data_offset; // minus padding
    const unsigned wi_block_data_offset = wo_block_data_offset; // minus padding

    // divide thread work
    constexpr unsigned NThreadWork = (NPerBlock + NPerThread - 1) / NPerThread;
    constexpr unsigned KThreadWork = (KPerBlock + KPerThread - 1) / KPerThread;
    constexpr unsigned YThreadWork = YPerBlock;
    constexpr unsigned XThreadWork = XPerBlock;

    const unsigned thread_id = threadIdx.x;

    itmp                            = thread_id;
    const unsigned n_thread_work_id = itmp / (KThreadWork * YThreadWork * XThreadWork);
    itmp -= n_thread_work_id * (KThreadWork * YThreadWork * XThreadWork);
    const unsigned k_thread_work_id = itmp / (YThreadWork * XThreadWork);
    itmp -= k_thread_work_id * (YThreadWork * XThreadWork);
    const unsigned y_thread_work_id = itmp / XThreadWork;
    const unsigned x_thread_work_id = itmp - y_thread_work_id * XThreadWork;

    const unsigned n_thread_data_offset  = n_thread_work_id * NPerThread;
    const unsigned k_thread_data_offset  = k_thread_work_id * KPerThread;
    const unsigned ho_thread_data_offset = y_thread_work_id * OutTileSizeH;
    const unsigned wo_thread_data_offset = x_thread_work_id * OutTileSizeW;

    const unsigned hi_thread_data_offset = ho_thread_data_offset;
    const unsigned wi_thread_data_offset = wo_thread_data_offset;

    // op
    auto f_set0 = [](TFloat& v) { v = TFloat(0); };
    auto f_copy = [](const TFloat& src, TFloat& dst) { dst = src; };

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(in_global_desc, "gridwise_convolution:  in_global_desc: ");
        print_ConstantTensorDescriptor(wei_global_desc, "gridwise_convolution: wei_global_desc: ");
        print_ConstantTensorDescriptor(out_global_desc, "gridwise_convolution: out_global_desc: ");
    }

    printf("threadIdx.x %u \t"
           "n_thread_data_offset %u, k_thread_data_offset %u, ho_thread_data_offset %u, "
           "wo_thread_data_offset %u\n",
           threadIdx.x,
           n_thread_data_offset,
           k_thread_data_offset,
           ho_thread_data_offset,
           wo_thread_data_offset);
#endif

    // set threadwise output tensor to 0
    threadwise_4d_tensor_op_unary<TFloat, decltype(out_thread_desc), decltype(f_set0)>(
        out_thread_desc, p_out_thread, f_set0);

    for(unsigned c_block_data_offset = 0; c_block_data_offset < in_global_desc.GetLength(I1);
        c_block_data_offset += CPerBlock, __syncthreads())
    {

#if 0
        if(threadIdx.x == 0)
        {
            printf("c_block_data_offset: %u\n", c_block_data_offset);
        }
#endif
        // copy input tensor to LDS
        blockwise_4d_tensor_op_binary<TFloat,
                                      decltype(in_block_global_desc),
                                      decltype(in_block_desc),
                                      NBlockOpLen0,
                                      NBlockOpLen1,
                                      NBlockOpLen2,
                                      NBlockOpLen3,
                                      decltype(f_copy),
                                      BlockSize>(
            in_block_global_desc,
            p_in_global + in_global_desc.Get1dIndex(n_block_data_offset,
                                                    c_block_data_offset,
                                                    hi_block_data_offset,
                                                    wi_block_data_offset),
            in_block_desc,
            p_in_block,
            f_copy);

        // copy weight tensor to LDS
        blockwise_4d_tensor_op_binary<TFloat,
                                      decltype(wei_block_global_desc),
                                      decltype(wei_block_desc),
                                      NBlockOpLen0,
                                      NBlockOpLen1,
                                      NBlockOpLen2,
                                      NBlockOpLen3,
                                      decltype(f_copy),
                                      BlockSize>(
            wei_block_global_desc,
            p_wei_global +
                wei_global_desc.Get1dIndex(k_block_data_offset, c_block_data_offset, 0, 0),
            wei_block_desc,
            p_wei_block,
            f_copy);

        __syncthreads();

        for(unsigned c_thread_data_offset = 0; c_thread_data_offset < CPerBlock;
            c_thread_data_offset += CPerThread)
        {

#if 0
            if(threadIdx.x == 0)
            {
                printf("c_thread_data_offset: %u\n", c_thread_data_offset);
            }
#endif
            // copy input tensor into register
            threadwise_4d_tensor_op_binary<TFloat,
                                           decltype(in_thread_block_desc),
                                           decltype(in_thread_desc),
                                           decltype(f_copy)>(
                in_thread_block_desc,
                p_in_block + in_block_desc.Get1dIndex(n_thread_data_offset,
                                                      c_thread_data_offset,
                                                      hi_thread_data_offset,
                                                      wi_thread_data_offset),
                in_thread_desc,
                p_in_thread,
                f_copy);

            // copy weight tensor into register
            threadwise_4d_tensor_op_binary<TFloat,
                                           decltype(wei_thread_block_desc),
                                           decltype(wei_thread_desc),
                                           decltype(f_copy)>(
                wei_thread_block_desc,
                p_wei_block +
                    wei_block_desc.Get1dIndex(k_thread_data_offset, c_thread_data_offset, 0, 0),
                wei_thread_desc,
                p_wei_thread,
                f_copy);

            // threadwise convolution
            threadwise_direct_convolution<TFloat,
                                          decltype(in_thread_desc),
                                          decltype(wei_thread_desc),
                                          decltype(out_thread_desc)>(in_thread_desc,
                                                                     p_in_thread,
                                                                     wei_thread_desc,
                                                                     p_wei_thread,
                                                                     out_thread_desc,
                                                                     p_out_thread);
        }
    }

    // copy output tensor from register to global mem
    threadwise_4d_tensor_op_binary<TFloat,
                                   decltype(out_thread_desc),
                                   decltype(out_thread_global_desc),
                                   decltype(f_copy)>(
        out_thread_desc,
        p_out_thread,
        out_thread_global_desc,
        p_out_global + out_global_desc.Get1dIndex(n_block_data_offset + n_thread_data_offset,
                                                  k_block_data_offset + k_thread_data_offset,
                                                  ho_block_data_offset + ho_thread_data_offset,
                                                  wo_block_data_offset + wo_thread_data_offset),
        f_copy);
}