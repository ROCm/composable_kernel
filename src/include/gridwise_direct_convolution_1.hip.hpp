#pragma once
#include "common.hip.hpp"
#include "ConstantTensorDescriptor.hip.hpp"
#include "blockwise_4d_tensor_op.hip.hpp"
#include "blockwise_direct_convolution.hip.hpp"

template <class Float,
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
          unsigned WoPerThread,
          unsigned BlockSize,
          unsigned GridSize>
__global__ void gridwise_direct_convolution_1(const Float* const __restrict__ p_in_global,
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

    constexpr unsigned Y = wei_global_desc.GetLength(I2);
    constexpr unsigned X = wei_global_desc.GetLength(I3);

    constexpr unsigned HiPerBlock = HoPerBlock + Y - 1;
    constexpr unsigned WiPerBlock = WoPerBlock + X - 1;

    constexpr unsigned NBlockWork = (out_global_desc.GetLength(I0) + NPerBlock - 1) / NPerBlock;
    constexpr unsigned KBlockWork = (out_global_desc.GetLength(I1) + KPerBlock - 1) / KPerBlock;
    constexpr unsigned HBlockWork = (out_global_desc.GetLength(I2) + HoPerBlock - 1) / HoPerBlock;
    constexpr unsigned WBlockWork = (out_global_desc.GetLength(I3) + WoPerBlock - 1) / WoPerBlock;

    constexpr auto in_block_global_desc = make_ConstantTensorDescriptor(
        Sequence<NPerBlock, CPerBlock, HiPerBlock, WiPerBlock>{}, in_global_desc.GetStrides());

    constexpr auto wei_block_global_desc = make_ConstantTensorDescriptor(
        Sequence<KPerBlock, CPerBlock, Y, X>{}, wei_global_desc.GetStrides());

    constexpr auto out_block_global_desc = make_ConstantTensorDescriptor(
        Sequence<NPerBlock, KPerBlock, HoPerBlock, WoPerBlock>{}, out_global_desc.GetStrides());

    constexpr auto in_block_desc = make_ConstantTensorDescriptor(in_block_global_desc.GetLengths());
    constexpr auto wei_block_desc =
        make_ConstantTensorDescriptor(wei_block_global_desc.GetLengths());
    constexpr auto out_block_desc =
        make_ConstantTensorDescriptor(out_block_global_desc.GetLengths());

    constexpr unsigned in_block_size  = in_block_desc.GetElementSpace();
    constexpr unsigned wei_block_size = wei_block_desc.GetElementSpace();
    constexpr unsigned out_block_size = out_block_desc.GetElementSpace();

    __shared__ Float p_in_block[in_block_size];
    __shared__ Float p_wei_block[wei_block_size];
    __shared__ Float p_out_block[out_block_size];

    const unsigned block_id = blockIdx.x;

    unsigned itmp            = block_id;
    unsigned n_block_work_id = itmp / (KBlockWork * HBlockWork * WBlockWork);
    itmp -= n_block_work_id * (KBlockWork * HBlockWork * WBlockWork);
    unsigned k_block_work_id = itmp / (HBlockWork * WBlockWork);
    itmp -= k_block_work_id * (HBlockWork * WBlockWork);
    unsigned h_block_work_id = itmp / WBlockWork;
    unsigned w_block_work_id = itmp - h_block_work_id * WBlockWork;

    unsigned n_block_work_begin  = n_block_work_id * NPerBlock;
    unsigned k_block_work_begin  = k_block_work_id * KPerBlock;
    unsigned ho_block_work_begin = h_block_work_id * HoPerBlock;
    unsigned wo_block_work_begin = w_block_work_id * WoPerBlock;

    unsigned hi_block_work_begin = ho_block_work_begin; // minus padding
    unsigned wi_block_work_begin = wo_block_work_begin; // minus padding

    constexpr auto blockwise_in_copy =
        Blockwise4dTensorCopy1<BlockSize,
                               Float,
                               decltype(in_block_global_desc),
                               decltype(in_block_desc),
                               decltype(in_block_desc.GetLengths())>{};

    constexpr auto blockwise_wei_copy =
        Blockwise4dTensorCopy1<BlockSize,
                               Float,
                               decltype(wei_block_global_desc),
                               decltype(wei_block_desc),
                               decltype(wei_block_desc.GetLengths())>{};

    constexpr auto blockwise_out_copy =
        Blockwise4dTensorCopy1<BlockSize,
                               Float,
                               decltype(out_block_desc),
                               decltype(out_block_global_desc),
                               decltype(out_block_desc.GetLengths())>{};

    // set output tensor in LDS to 0
    blockwise_4d_tensor_set_zero<BlockSize>(out_block_desc, p_out_block);

    for(unsigned c_block_work_begin = 0; c_block_work_begin < in_global_desc.GetLength(I1);
        c_block_work_begin += CPerBlock)
    {
        // copy input tensor to LDS
        blockwise_in_copy.Run(p_in_global + in_global_desc.Get1dIndex(n_block_work_begin,
                                                                      c_block_work_begin,
                                                                      hi_block_work_begin,
                                                                      wi_block_work_begin),
                              p_in_block);

        // copy weight tensor to LDS
        blockwise_wei_copy.Run(
            p_wei_global + wei_global_desc.Get1dIndex(k_block_work_begin, c_block_work_begin, 0, 0),
            p_wei_block);

        __syncthreads();

        // blockwise convolution
        blockwise_direct_convolution<BlockSize,
                                     Float,
                                     decltype(in_block_desc),
                                     decltype(wei_block_desc),
                                     decltype(out_block_desc),
                                     NPerThread,
                                     KPerThread,
                                     CPerThread,
                                     HoPerThread,
                                     WoPerThread>(
            in_block_desc, p_in_block, wei_block_desc, p_wei_block, out_block_desc, p_out_block);

        __syncthreads();
    }

    // copy output tensor from LDS to device mem
    blockwise_out_copy.Run(p_out_block,
                           p_out_global + out_global_desc.Get1dIndex(n_block_work_begin,
                                                                     k_block_work_begin,
                                                                     ho_block_work_begin,
                                                                     wo_block_work_begin));
}
