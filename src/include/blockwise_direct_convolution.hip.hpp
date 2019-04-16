#pragma once
#include "ConstantTensorDescriptor.hip.hpp"
#include "threadwise_4d_tensor_op.hip.hpp"
#include "threadwise_direct_convolution.hip.hpp"

template <index_t BlockSize,
          class Float,
          class InBlockDesc,
          class WeiBlockDesc,
          class OutBlockDesc,
          index_t NPerThread,
          index_t KPerThread,
          index_t CPerThread,
          index_t HoPerThread,
          index_t WoPerThread>
__device__ void blockwise_direct_convolution(InBlockDesc,
                                             Float* const __restrict__ p_in_block,
                                             WeiBlockDesc,
                                             Float* const __restrict__ p_wei_block,
                                             OutBlockDesc,
                                             Float* __restrict__ p_out_block)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_block_desc  = InBlockDesc{};
    constexpr auto wei_block_desc = WeiBlockDesc{};
    constexpr auto out_block_desc = OutBlockDesc{};

    constexpr index_t Y = wei_block_desc.GetLength(I2);
    constexpr index_t X = wei_block_desc.GetLength(I3);

    constexpr index_t InTileSizeH = HoPerThread + Y - 1;
    constexpr index_t InTileSizeW = WoPerThread + X - 1;

    // divide thread work
    constexpr index_t NThreadWork = (out_block_desc.GetLength(I0) + NPerThread - 1) / NPerThread;
    constexpr index_t KThreadWork = (out_block_desc.GetLength(I1) + KPerThread - 1) / KPerThread;
    constexpr index_t YThreadWork = (out_block_desc.GetLength(I2) + HoPerThread - 1) / HoPerThread;
    constexpr index_t XThreadWork = (out_block_desc.GetLength(I3) + WoPerThread - 1) / WoPerThread;

#if 0
    if(get_thread_local_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(in_block_desc);
        print_ConstantTensorDescriptor(wei_block_desc);
        print_ConstantTensorDescriptor(out_block_desc);
    }
#endif

    constexpr auto in_thread_desc =
        make_ConstantTensorDescriptor(Sequence<NPerThread, CPerThread, InTileSizeH, InTileSizeW>{});

    constexpr auto wei_thread_desc =
        make_ConstantTensorDescriptor(Sequence<KPerThread, CPerThread, Y, X>{});

    constexpr auto out_thread_desc =
        get_convolution_output_default_4d_tensor_descriptor(in_thread_desc, wei_thread_desc);

    constexpr auto in_thread_block_desc =
        make_ConstantTensorDescriptor(in_thread_desc.GetLengths(), in_block_desc.GetStrides());

    constexpr auto wei_thread_block_desc =
        make_ConstantTensorDescriptor(wei_thread_desc.GetLengths(), wei_block_desc.GetStrides());

    constexpr auto out_thread_block_desc =
        make_ConstantTensorDescriptor(out_thread_desc.GetLengths(), out_block_desc.GetStrides());

    const index_t thread_id = get_thread_local_1d_id();

    for(index_t thread_work_id = thread_id;
        thread_work_id < NThreadWork * KThreadWork * YThreadWork * XThreadWork;
        thread_work_id += BlockSize)
    {
        index_t itmp             = thread_work_id;
        index_t n_thread_work_id = itmp / (KThreadWork * YThreadWork * XThreadWork);
        itmp -= n_thread_work_id * (KThreadWork * YThreadWork * XThreadWork);
        index_t k_thread_work_id = itmp / (YThreadWork * XThreadWork);
        itmp -= k_thread_work_id * (YThreadWork * XThreadWork);
        index_t y_thread_work_id = itmp / XThreadWork;
        index_t x_thread_work_id = itmp - y_thread_work_id * XThreadWork;

        index_t n_thread_data_begin  = n_thread_work_id * NPerThread;
        index_t k_thread_data_begin  = k_thread_work_id * KPerThread;
        index_t ho_thread_data_begin = y_thread_work_id * HoPerThread;
        index_t wo_thread_data_begin = x_thread_work_id * WoPerThread;

        index_t hi_thread_data_begin = ho_thread_data_begin; // minus padding
        index_t wi_thread_data_begin = wo_thread_data_begin; // minus padding

        Float p_out_thread[out_thread_desc.GetElementSpace()];

        threadwise_4d_tensor_copy(out_block_desc,
                                  p_out_block + out_block_desc.Get1dIndex(n_thread_data_begin,
                                                                          k_thread_data_begin,
                                                                          ho_thread_data_begin,
                                                                          wo_thread_data_begin),
                                  out_thread_desc,
                                  p_out_thread,
                                  out_thread_desc.GetLengths());

        for(index_t c_thread_data_begin = 0; c_thread_data_begin < in_block_desc.GetLength(I1);
            c_thread_data_begin += CPerThread)
        {
            // threadwise convolution
            threadwise_direct_convolution_2(
                in_thread_block_desc,
                p_in_block + in_block_desc.Get1dIndex(n_thread_data_begin,
                                                      c_thread_data_begin,
                                                      hi_thread_data_begin,
                                                      wi_thread_data_begin),
                wei_thread_block_desc,
                p_wei_block +
                    wei_block_desc.Get1dIndex(k_thread_data_begin, c_thread_data_begin, 0, 0),
                out_thread_desc,
                p_out_thread);
        }

        // copy output into LDS
        threadwise_4d_tensor_copy(out_thread_desc,
                                  p_out_thread,
                                  out_block_desc,
                                  p_out_block + out_block_desc.Get1dIndex(n_thread_data_begin,
                                                                          k_thread_data_begin,
                                                                          ho_thread_data_begin,
                                                                          wo_thread_data_begin),
                                  out_thread_desc.GetLengths());
    }
}
