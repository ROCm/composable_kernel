#pragma once
#include "ConstantTensorDescriptor.hip.hpp"
#include "blockwise_winograd_transform.hip.hpp"
#include "threadwise_winograd_transform.hip.hpp"

template <class Float,
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
          unsigned BlockSize,
          unsigned GridSize>
__global__ void gridwise_winograd_convolution(const Float* const __restrict__ p_in_global,
                                              const Float* const __restrict__ p_wei_global,
                                              Float* const __restrict__ p_out_global)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_global_desc  = InGlobalDesc{};
    constexpr auto wei_global_desc = WeiGlobalDesc{};
    constexpr auto out_global_desc = OutGlobalDesc{};

    constexpr unsigned S = wei_global_desc.GetLength(I2);
    constexpr unsigned R = wei_global_desc.GetLength(I3);

    constexpr unsigned HoPerBlock = OutTileSizeH * YPerBlock;
    constexpr unsigned WoPerBlock = OutTileSizeW * XPerBlock;

    constexpr unsigned HiPerBlock = HoPerBlock + S - 1;
    constexpr unsigned WiPerBlock = WoPerBlock + R - 1;

    constexpr unsigned InTileSizeH = OutTileSizeH + S - 1;
    constexpr unsigned InTileSizeW = OutTileSizeW + R - 1;

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

    const unsigned n_block_data_begin = n_block_work_id * NPerBlock;
    const unsigned k_block_data_begin = k_block_work_id * KPerBlock;
    const unsigned y_block_data_begin = y_block_work_id * YPerBlock;
    const unsigned x_block_data_begin = x_block_work_id * XPerBlock;

    const unsigned ho_block_data_begin = y_block_data_begin * OutTileSizeH;
    const unsigned wo_block_data_begin = x_block_data_begin * OutTileSizeW;

    const unsigned hi_block_data_begin = ho_block_data_begin; // minus padding
    const unsigned wi_block_data_begin = wo_block_data_begin; // minus padding

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

    const unsigned n_thread_data_begin = n_thread_work_id * NPerThread;
    const unsigned k_thread_data_begin = k_thread_work_id * KPerThread;
    const unsigned y_thread_data_begin = y_thread_work_id;
    const unsigned x_thread_data_begin = x_thread_work_id;

    // block data
    constexpr auto in_transform_block_desc = make_ConstantTensorDescriptor(
        Sequence<NPerBlock, CPerBlock, YPerBlock * InTileSizeH, XPerBlock * InTileSizeW>{});

    constexpr auto wei_transform_block_desc =
        make_ConstantTensorDescriptor(Sequence<KPerBlock, CPerBlock, InTileSizeH, InTileSizeW>{});

    __shared__ Float p_in_transform_block[in_transform_block_desc.GetElementSpace()];
    __shared__ Float p_wei_transform_block[wei_transform_block_desc.GetElementSpace()];

    // thread data
    constexpr auto in_transform_thread_block_desc =
        make_ConstantTensorDescriptor(Sequence<NPerThread, CPerThread, InTileSizeH, InTileSizeW>{},
                                      in_transform_block_desc.GetStrides());

    constexpr auto wei_transform_thread_block_desc =
        make_ConstantTensorDescriptor(Sequence<KPerThread, CPerThread, InTileSizeH, InTileSizeW>{},
                                      wei_transform_block_desc.GetStrides());

    constexpr auto out_transform_thread_desc =
        make_ConstantTensorDescriptor(Sequence<NPerThread, KPerThread, InTileSizeH, InTileSizeW>{});

    constexpr auto out_thread_desc = make_ConstantTensorDescriptor(
        Sequence<NPerThread, KPerThread, OutTileSizeH, OutTileSizeW>{});

    constexpr auto out_thread_global_desc =
        make_ConstantTensorDescriptor(out_thread_desc.GetLengths(), out_global_desc.GetStrides());

    Float p_out_transform_thread[out_transform_thread_desc.GetElementSpace()];
    Float p_out_thread[out_thread_desc.GetElementSpace()];

#if 0
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("in_transform_block_size %u, wei_transform_block_size %u, out_transform_thread_size "
               "%u, out_thread_size %u \n",
               in_transform_block_size,
               wei_transform_block_size,
               out_transform_thread_size,
               out_thread_size);
    }
#endif

    // set threadwise output transform tensor to 0
    threadwise_4d_tensor_set_zero(out_transform_thread_desc, p_out_transform_thread);

    for(unsigned c_block_data_begin = 0; c_block_data_begin < in_global_desc.GetLength(I1);
        c_block_data_begin += CPerBlock, __syncthreads())
    {
#if 0
        // blockwise transform input
        blockwise_winograd_transform_input<Float,
                                           InTileSizeH,
                                           InTileSizeW,
                                           S,
                                           R,
                                           OutTileSizeH,
                                           OutTileSizeW,
                                           NPerBlock,
                                           CPerBlock,
                                           YPerBlock,
                                           XPerBlock,
                                           BlockSize>(
            p_in_global + in_global_desc.Get1dIndex(n_block_data_begin,
                                                    c_block_data_begin,
                                                    hi_block_data_begin,
                                                    wi_block_data_begin),
            p_in_transform_block);

#endif
        // blockwise transform weights
        blockwise_winograd_transform_weight<Float,
                                            InTileSizeH,
                                            InTileSizeW,
                                            S,
                                            R,
                                            OutTileSizeH,
                                            OutTileSizeW,
                                            KPerBlock,
                                            CPerBlock,
                                            BlockSize>(
            p_wei_global + wei_global_desc.Get1dIndex(k_block_data_begin, c_block_data_begin, 0, 0),
            p_wei_transform_block);

        for(unsigned c_thread_data = 0; c_thread_data < CPerBlock; c_thread_data += CPerThread)
        {
            // threadwise point multiplication
            threadwise_winograd_calculate_transformed_output<
                Float,
                decltype(in_transform_thread_block_desc),
                decltype(wei_transform_thread_block_desc),
                decltype(out_transform_thread_desc),
                InTileSizeH,
                InTileSizeW,
                S,
                R,
                OutTileSizeH,
                OutTileSizeW>(in_transform_thread_block_desc,
                              p_in_transform_block + in_transform_block_desc.Get1dIndex(
                                                         n_thread_data_begin,
                                                         c_thread_data,
                                                         y_thread_data_begin * InTileSizeH,
                                                         x_thread_data_begin * InTileSizeW),
                              wei_transform_thread_block_desc,
                              p_wei_transform_block + wei_transform_block_desc.Get1dIndex(
                                                          k_thread_data_begin, c_thread_data, 0, 0),
                              out_transform_thread_desc,
                              p_out_transform_thread);
        }
    };

    // transform back
    threadwise_winograd_reverse_transform_output<Float,
                                                 decltype(out_transform_thread_desc),
                                                 decltype(out_thread_desc),
                                                 InTileSizeH,
                                                 InTileSizeW,
                                                 S,
                                                 R,
                                                 OutTileSizeH,
                                                 OutTileSizeW>(
        out_transform_thread_desc, p_out_transform_thread, out_thread_desc, p_out_thread);

    // copy output tensor from register to global mem
    threadwise_4d_tensor_copy(
        out_thread_desc,
        p_out_thread,
        out_thread_global_desc,
        p_out_global +
            out_global_desc.Get1dIndex(n_block_data_begin + n_thread_data_begin,
                                       k_block_data_begin + k_thread_data_begin,
                                       ho_block_data_begin + y_thread_data_begin * OutTileSizeH,
                                       wo_block_data_begin + x_thread_data_begin * OutTileSizeW));
}
