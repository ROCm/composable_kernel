#pragma once
#include "constant_tensor_descriptor.cuh"
#include "threadwise_tensor_op.cuh"
#include "threadwise_direct_convolution.cuh"

template <class TFloat,
          class InBlockDesc,
          class WeiBlockDesc,
          class OutBlockDesc,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned NPerThread,
          unsigned KPerThread,
          unsigned CPerThread,
          unsigned BlockSize>
__device__ void blockwise_direct_convolution(InBlockDesc,
                                             TFloat* const __restrict__ p_in_block,
                                             WeiBlockDesc,
                                             TFloat* const __restrict__ p_wei_block,
                                             OutBlockDesc,
                                             TFloat* __restrict__ p_out_block)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_block_desc  = InBlockDesc{};
    constexpr auto wei_block_desc = WeiBlockDesc{};
    constexpr auto out_block_desc = OutBlockDesc{};

    constexpr unsigned S = wei_block_desc.GetLength(I2);
    constexpr unsigned R = wei_block_desc.GetLength(I3);

    constexpr unsigned InTileSizeH = OutTileSizeH + S - 1;
    constexpr unsigned InTileSizeW = OutTileSizeW + R - 1;

    // divide thread work
    constexpr unsigned NThreadWork = (out_block_desc.GetLength(I0) + NPerThread - 1) / NPerThread;
    constexpr unsigned KThreadWork = (out_block_desc.GetLength(I1) + KPerThread - 1) / KPerThread;
    constexpr unsigned YThreadWork =
        (out_block_desc.GetLength(I2) + OutTileSizeH - 1) / OutTileSizeH;
    constexpr unsigned XThreadWork =
        (out_block_desc.GetLength(I3) + OutTileSizeW - 1) / OutTileSizeW;

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(in_block_desc);
        print_ConstantTensorDescriptor(wei_block_desc);
        print_ConstantTensorDescriptor(out_block_desc);
    }
#endif

    constexpr auto in_thread_desc =
        make_ConstantTensorDescriptor(Sequence<NPerThread, CPerThread, InTileSizeH, InTileSizeW>{});

    constexpr auto wei_thread_desc =
        make_ConstantTensorDescriptor(Sequence<KPerThread, CPerThread, S, R>{});

    constexpr auto out_thread_desc = make_ConstantTensorDescriptor(
        Sequence<NPerThread, KPerThread, OutTileSizeH, OutTileSizeW>{});

    constexpr auto in_thread_block_desc =
        make_ConstantTensorDescriptor(in_thread_desc.GetLengths(), in_block_desc.GetStrides());

    constexpr auto wei_thread_block_desc =
        make_ConstantTensorDescriptor(wei_thread_desc.GetLengths(), wei_block_desc.GetStrides());

    constexpr auto out_thread_block_desc =
        make_ConstantTensorDescriptor(out_thread_desc.GetLengths(), out_block_desc.GetStrides());

    const unsigned thread_id = threadIdx.x;

    for(unsigned thread_work_id = thread_id;
        thread_work_id < NThreadWork * KThreadWork * YThreadWork * XThreadWork;
        thread_work_id += BlockSize)
    {
        unsigned itmp             = thread_work_id;
        unsigned n_thread_work_id = itmp / (KThreadWork * YThreadWork * XThreadWork);
        itmp -= n_thread_work_id * (KThreadWork * YThreadWork * XThreadWork);
        unsigned k_thread_work_id = itmp / (YThreadWork * XThreadWork);
        itmp -= k_thread_work_id * (YThreadWork * XThreadWork);
        unsigned y_thread_work_id = itmp / XThreadWork;
        unsigned x_thread_work_id = itmp - y_thread_work_id * XThreadWork;

        unsigned n_thread_data_begin  = n_thread_work_id * NPerThread;
        unsigned k_thread_data_begin  = k_thread_work_id * KPerThread;
        unsigned ho_thread_data_begin = y_thread_work_id * OutTileSizeH;
        unsigned wo_thread_data_begin = x_thread_work_id * OutTileSizeW;

        unsigned hi_thread_data_begin = ho_thread_data_begin; // minus padding
        unsigned wi_thread_data_begin = wo_thread_data_begin; // minus padding

        TFloat p_in_thread[in_thread_desc.GetElementSpace()];
        TFloat p_wei_thread[wei_thread_desc.GetElementSpace()];
        TFloat p_out_thread[out_thread_desc.GetElementSpace()];

        threadwise_4d_tensor_copy(out_thread_block_desc,
                                  p_out_block + out_block_desc.Get1dIndex(n_thread_data_begin,
                                                                          k_thread_data_begin,
                                                                          ho_thread_data_begin,
                                                                          wo_thread_data_begin),
                                  out_thread_desc,
                                  p_out_thread);

        for(unsigned c_thread_data_begin = 0; c_thread_data_begin < in_block_desc.GetLength(I1);
            c_thread_data_begin += CPerThread)
        {
            // copy input into register
            threadwise_4d_tensor_copy(in_thread_block_desc,
                                      p_in_block + in_block_desc.Get1dIndex(n_thread_data_begin,
                                                                            c_thread_data_begin,
                                                                            hi_thread_data_begin,
                                                                            wi_thread_data_begin),
                                      in_thread_desc,
                                      p_in_thread);

            // copy weight into register
            threadwise_4d_tensor_copy(
                wei_thread_block_desc,
                p_wei_block +
                    wei_block_desc.Get1dIndex(k_thread_data_begin, c_thread_data_begin, 0, 0),
                wei_thread_desc,
                p_wei_thread);

            // threadwise convolution
            threadwise_direct_convolution(in_thread_desc,
                                          p_in_thread,
                                          wei_thread_desc,
                                          p_wei_thread,
                                          out_thread_desc,
                                          p_out_thread);
        }

        // copy output into LDS
        threadwise_4d_tensor_copy(out_thread_desc,
                                  p_out_thread,
                                  out_thread_block_desc,
                                  p_out_block + out_block_desc.Get1dIndex(n_thread_data_begin,
                                                                          k_thread_data_begin,
                                                                          ho_thread_data_begin,
                                                                          wo_thread_data_begin));
    }
}
