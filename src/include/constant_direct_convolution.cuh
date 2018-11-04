#pragma once
#include "constant_tensor_descriptor.cuh"

template <class TFloat,
          class SrcDesc,
          class DstDesc,
          unsigned NWorkLen0,
          unsigned NWorkLen1,
          unsigned NWorkLen2,
          unsigned NWorkLen3,
          class F>
__device__ void blockwise_4d_tensor_op(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

#if 1
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(src_desc);
        print_ConstantTensorDescriptor(dst_desc);
    }
#endif

    constexpr unsigned NWorkStride3 = 1;
    constexpr unsigned NWorkStride2 = NWorkLen3 * NWorkStride3;
    constexpr unsigned NWorkStride1 = NWorkLen2 * NWorkStride2;
    constexpr unsigned NWorkStride0 = NWorkLen1 * NWorkStride1;

    unsigned itmp =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.y * blockDim.x);

    const unsigned did0_begin = itmp / NWorkStride0;

    itmp -= did0_begin * NWorkStride0;

    const unsigned did1_begin = itmp / NWorkStride1;

    itmp -= did1_begin * NWorkStride1;

    const unsigned did2_begin = itmp / NWorkStride2;

    itmp -= did2_begin * NWorkStride2;

    const unsigned did3_begin = itmp / NWorkStride3;

    for(unsigned did0 = did0_begin; did0 < src_desc.GetLength(I0); did0 += NWorkLen0)
    {
        for(unsigned did1 = did1_begin; did1 < src_desc.GetLength(I1); did1 += NWorkLen1)
        {
            for(unsigned did2 = did2_begin; did2 < src_desc.GetLength(I2); did2 += NWorkLen2)
            {
                for(unsigned did3 = did3_begin; did3 < src_desc.GetLength(I3); did3 += NWorkLen3)
                {
                    const unsigned sindex =
                        src_desc.GetStride(I0) * did0 + src_desc.GetStride(I1) * did1 +
                        src_desc.GetStride(I2) * did2 + src_desc.GetStride(I3) * did3;

                    const unsigned dindex =
                        dst_desc.GetStride(I0) * did0 + dst_desc.GetStride(I1) * did1 +
                        dst_desc.GetStride(I2) * did2 + dst_desc.GetStride(I3) * did3;

                    f(p_src[dindex], p_dst[sindex]);

#if 0
                    // if(threadIdx.x == 0)
                    {
                        printf("blockwise_4d_tensor_op: 1: thread id %u, \t"
                               "sindex %u, p_src[sindex] %f, \t"
                               "dindex %u, p_dst[dindex] %f\n",
                               threadIdx.x,
                               sindex,
                               p_src[sindex],
                               dindex,
                               p_dst[dindex]);
                    }
#endif
                }
            }
        }
    }
}

template <class TFloat, class SrcDesc, class DstDesc, class F>
__device__ void threadwise_4d_tensor_op(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

#if 1
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(src_desc);
        print_ConstantTensorDescriptor(dst_desc);
    }
#endif

    for(unsigned did0 = 0; did0 < src_desc.GetLength(I0); ++did0)
    {
        for(unsigned did1 = 0; did1 < src_desc.GetLength(I1); ++did1)
        {
            for(unsigned did2 = 0; did2 < src_desc.GetLength(I2); ++did2)
            {
                for(unsigned did3 = 0; did3 < src_desc.GetLength(I3); ++did3)
                {
                    const unsigned sindex =
                        src_desc.GetStride(I0) * did0 + src_desc.GetStride(I1) * did1 +
                        src_desc.GetStride(I2) * did2 + src_desc.GetStride(I3) * did3;

                    const unsigned dindex =
                        dst_desc.GetStride(I0) * did0 + dst_desc.GetStride(I1) * did1 +
                        dst_desc.GetStride(I2) * did2 + dst_desc.GetStride(I3) * did3;

                    f(p_src[sindex], p_dst[dindex]);

#if 0
                    if(threadIdx.x == 0)
                    {
                        printf("threadwise_4d_tensor_op: 1: thread id %u, \t"
                               "sindex %u, p_src[sindex] %f, \t"
                               "dindex %u, p_dst[dindex] %f\n",
                               threadIdx.x,
                               sindex,
                               p_src[sindex],
                               dindex,
                               p_dst[dindex]);
                    }
#endif
                }
            }
        }
    }
}

template <class TFloat, class InDesc, class WeiDesc, class OutDesc>
__device__ void threadwise_direct_convolution(InDesc,
                                              TFloat* const __restrict__ p_in,
                                              WeiDesc,
                                              TFloat* const __restrict__ p_wei,
                                              OutDesc,
                                              TFloat* __restrict__ p_out)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

#if 1
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(in_desc);
        print_ConstantTensorDescriptor(wei_desc);
        print_ConstantTensorDescriptor(out_desc);
    }
#endif

    for(unsigned n = 0; n < out_desc.GetLength(I0); ++n)
    {
        for(unsigned k = 0; k < out_desc.GetLength(I1); ++k)
        {
            for(unsigned ho = 0; ho < out_desc.GetLength(I2); ++ho)
            {
                for(unsigned wo = 0; wo < out_desc.GetLength(I3); ++wo)
                {
                    for(unsigned c = 0; c < wei_desc.GetLength(I1); ++c)
                    {
                        for(unsigned s = 0; s < wei_desc.GetLength(I2); ++s)
                        {
                            for(unsigned r = 0; r < wei_desc.GetLength(I3); ++r)
                            {
                                const unsigned hi = ho + s;
                                const unsigned wi = wo + r;

                                const unsigned in_index =
                                    in_desc.GetStride(I0) * n + in_desc.GetStride(I1) * c +
                                    in_desc.GetStride(I2) * hi + in_desc.GetStride(I3) * wi;

                                const unsigned wei_index =
                                    wei_desc.GetStride(I0) * k + wei_desc.GetStride(I1) * c +
                                    wei_desc.GetStride(I2) * s + in_desc.GetStride(I3) * r;

                                const unsigned out_index =
                                    out_desc.GetStride(I0) * n + out_desc.GetStride(I1) * k +
                                    out_desc.GetStride(I2) * ho + out_desc.GetStride(I3) * wo;

                                p_out[out_index] += p_wei[wei_index] * p_in[in_index];

#if 0
                                if(threadIdx.x == 0)
                                {
                                    printf("threadwise_direct_convolution: 1: \t"
                                           "threadIdx.x %u\t"
                                           "out_index %u, p_out[out_index] %f, \t"
                                           "wei_index %u, p_wei[wei_index] %f, \t"
                                           "in_index %u, p_in[in_index] %f\n",
                                           threadIdx.x,
                                           out_index,
                                           p_out[out_index],
                                           wei_index,
                                           p_wei[wei_index],
                                           in_index,
                                           p_in[in_index]);
                                }
#endif
                            }
                        }
                    }
                }
            }
        }
    }
}

template <class TFloat,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          unsigned S,
          unsigned R,
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned NPerBlock,
          unsigned KPerBlock,
          unsigned YPerBlock,
          unsigned XPerBlock,
          unsigned CPerBlockLoop>
__device__ void blockwise_convolution(InDesc,
                                      TFloat* const __restrict__ p_in,
                                      WeiDesc,
                                      TFloat* const __restrict__ p_wei,
                                      OutDesc,
                                      TFloat* __restrict__ p_out)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

#if 1
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(in_desc);
        print_ConstantTensorDescriptor(wei_desc);
        print_ConstantTensorDescriptor(out_desc);
    }
#endif

    constexpr auto in_thread_src_desc = make_ConstantTensorDescriptor(
        Sequence<1, CPerBlockLoop, OutTileSizeH + S - 1, OutTileSizeW + R - 1>{},
        in_desc.GetStrides());

    constexpr auto wei_thread_src_desc =
        make_ConstantTensorDescriptor(Sequence<1, CPerBlockLoop, S, R>{}, wei_desc.GetStrides());

    constexpr auto out_thread_src_desc = make_ConstantTensorDescriptor(
        Sequence<1, 1, OutTileSizeH, OutTileSizeW>{}, out_desc.GetStrides());

    constexpr auto in_thread_dst_desc =
        make_ConstantTensorDescriptor(in_thread_src_desc.GetLengths());

    constexpr auto wei_thread_dst_desc =
        make_ConstantTensorDescriptor(wei_thread_src_desc.GetLengths());

    constexpr auto out_thread_dst_desc =
        make_ConstantTensorDescriptor(out_thread_src_desc.GetLengths());

    const unsigned thread_sz = blockDim.x * blockDim.y * blockDim.z;

    const unsigned thread_id =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.y * blockDim.x);

    for(unsigned thread_work_id = thread_id;
        thread_work_id < NPerBlock * KPerBlock * YPerBlock * XPerBlock;
        thread_work_id += thread_sz)
    {
        unsigned itmp             = thread_work_id;
        unsigned n_thread_work_id = itmp / (KPerBlock * YPerBlock * XPerBlock);
        itmp -= n_thread_work_id * (KPerBlock * YPerBlock * XPerBlock);
        unsigned k_thread_work_id = itmp / (YPerBlock * XPerBlock);
        itmp -= k_thread_work_id * (YPerBlock * XPerBlock);
        unsigned y_thread_work_id = itmp / XPerBlock;
        unsigned x_thread_work_id = itmp - y_thread_work_id * XPerBlock;

        unsigned n_thread_work_begin  = n_thread_work_id * 1;
        unsigned k_thread_work_begin  = k_thread_work_id * 1;
        unsigned ho_thread_work_begin = y_thread_work_id * OutTileSizeH;
        unsigned wo_thread_work_begin = x_thread_work_id * OutTileSizeW;

        unsigned hi_thread_work_begin = ho_thread_work_begin; // minus padding
        unsigned wi_thread_work_begin = wo_thread_work_begin; // minus padding

        TFloat p_in_thread[1 * CPerBlockLoop * InTileSizeH * InTileSizeW];
        TFloat p_wei_thread[1 * CPerBlockLoop * S * R];
        TFloat p_out_thread[1 * 1 * OutTileSizeH * OutTileSizeW];

        auto f_copy = [](const TFloat& src, TFloat& dst) { dst = src; };

        // copy input tensor into register
        threadwise_4d_tensor_op<TFloat,
                                decltype(in_thread_src_desc),
                                decltype(in_thread_dst_desc),
                                decltype(f_copy)>(
            in_thread_src_desc,
            p_in + in_desc.Get1dIndex(
                       n_thread_work_begin, 0, hi_thread_work_begin, wi_thread_work_begin),
            in_thread_dst_desc,
            p_in_thread,
            f_copy);

        // copy weight tensor into register
        threadwise_4d_tensor_op<TFloat,
                                decltype(wei_thread_src_desc),
                                decltype(wei_thread_dst_desc),
                                decltype(f_copy)>(
            wei_thread_src_desc,
            p_wei + wei_desc.Get1dIndex(k_thread_work_begin, 0, 0, 0),
            wei_thread_dst_desc,
            p_wei_thread,
            f_copy);

        // copy output tensor into register
        threadwise_4d_tensor_op<TFloat,
                                decltype(out_thread_src_desc),
                                decltype(out_thread_dst_desc),
                                decltype(f_copy)>(out_thread_src_desc,
                                                  p_out + out_desc.Get1dIndex(n_thread_work_begin,
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

        // accumulate output tensor into device mem
        threadwise_4d_tensor_op<TFloat,
                                decltype(out_thread_dst_desc),
                                decltype(out_thread_src_desc),
                                decltype(f_copy)>(out_thread_dst_desc,
                                                  p_out_thread,
                                                  out_thread_src_desc,
                                                  p_out + out_desc.Get1dIndex(n_thread_work_begin,
                                                                              k_thread_work_begin,
                                                                              ho_thread_work_begin,
                                                                              wo_thread_work_begin),
                                                  f_copy);
    }
}

template <class TFloat,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned NPerBlock,
          unsigned KPerBlock,
          unsigned YPerBlock,
          unsigned XPerBlock,
          unsigned CPerBlockLoop>
__global__ void gridwise_convolution(InDesc,
                                     TFloat* const __restrict__ p_in,
                                     WeiDesc,
                                     TFloat* const __restrict__ p_wei,
                                     OutDesc,
                                     TFloat* __restrict__ p_out)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

    constexpr unsigned S = wei_desc.GetLength(I2);
    constexpr unsigned R = wei_desc.GetLength(I3);

#if 1
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(in_desc);
        print_ConstantTensorDescriptor(wei_desc);
        print_ConstantTensorDescriptor(out_desc);
    }
#endif

    constexpr unsigned NBlockWork = (in_desc.GetLength(I0) + NPerBlock - 1) / NPerBlock;
    constexpr unsigned YBlockWork = (in_desc.GetLength(I2) + YPerBlock - 1) / YPerBlock;
    constexpr unsigned XBlockWork = (in_desc.GetLength(I3) + XPerBlock - 1) / XPerBlock;

    constexpr unsigned KBlockWork = (wei_desc.GetLength(I1) + KPerBlock - 1) / KPerBlock;

    const unsigned block_id =
        blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.y * gridDim.x);

    constexpr auto in_block_desc =
        make_ConstantTensorDescriptor(Sequence<NPerBlock,
                                               CPerBlockLoop,
                                               YPerBlock * OutTileSizeH + S - 1,
                                               XPerBlock * OutTileSizeW + R - 1>{});
    constexpr auto wei_block_desc =
        make_ConstantTensorDescriptor(Sequence<KPerBlock, CPerBlockLoop, S, R>{});

    constexpr auto out_block_desc = make_ConstantTensorDescriptor(
        Sequence<NPerBlock, KPerBlock, YPerBlock * OutTileSizeH, XPerBlock * OutTileSizeW>{});

    __shared__ TFloat p_in_block[NPerBlock * CPerBlockLoop * (YPerBlock * OutTileSizeH + S - 1) *
                                 (XPerBlock * OutTileSizeW + R - 1)];
    __shared__ TFloat p_wei_block[KPerBlock * CPerBlockLoop * S * R];
    __shared__ TFloat p_out_block[NPerBlock * KPerBlock * (YPerBlock * OutTileSizeH) *
                                  (XPerBlock * OutTileSizeW)];

    unsigned itmp            = block_id;
    unsigned n_block_work_id = itmp / (KBlockWork * YBlockWork * XBlockWork);
    itmp -= n_block_work_id * (KBlockWork * YBlockWork * XBlockWork);
    unsigned k_block_work_id = itmp / (YBlockWork * XBlockWork);
    itmp -= k_block_work_id * (YBlockWork * XBlockWork);
    unsigned y_block_work_id = itmp / XBlockWork;
    unsigned x_block_work_id = itmp - y_block_work_id * XBlockWork;

    unsigned n_block_work_begin = n_block_work_id * NPerBlock;
    unsigned k_block_work_begin = k_block_work_id * KPerBlock;
    unsigned y_block_work_begin = y_block_work_id * YPerBlock;
    unsigned x_block_work_begin = x_block_work_id * XPerBlock;

    unsigned ho_block_work_begin = y_block_work_begin * OutTileSizeH;
    unsigned wo_block_work_begin = x_block_work_begin * OutTileSizeW;

    unsigned hi_block_work_begin = ho_block_work_begin; // minus padding
    unsigned wi_block_work_begin = wo_block_work_begin; // minus padding

    for(unsigned c_block_work_begin = 0; c_block_work_begin < in_desc.GetLength(I1);
        c_block_work_begin += CPerBlockLoop)
    {
        auto f_copy = [](const TFloat& src, TFloat& dst) { dst = src; };

        // copy input tensor to LDS
        blockwise_4d_tensor_op<TFloat,
                               decltype(in_desc),
                               decltype(in_block_desc),
                               1,
                               1,
                               1,
                               64,
                               decltype(f_copy)>(in_desc,
                                                 p_in + in_desc.Get1dIndex(n_block_work_begin,
                                                                           c_block_work_begin,
                                                                           hi_block_work_begin,
                                                                           wi_block_work_begin),
                                                 in_block_desc,
                                                 p_in_block,
                                                 f_copy);

        // copy weight tensor to LDS
        blockwise_4d_tensor_op<TFloat,
                               decltype(wei_desc),
                               decltype(wei_block_desc),
                               1,
                               1,
                               1,
                               64,
                               decltype(f_copy)>(
            wei_desc,
            p_wei + wei_desc.Get1dIndex(k_block_work_begin, c_block_work_begin, 0, 0),
            wei_block_desc,
            p_wei_block,
            f_copy);

        // copy output tensor to LDS
        blockwise_4d_tensor_op<TFloat,
                               decltype(out_desc),
                               decltype(out_block_desc),
                               1,
                               1,
                               1,
                               64,
                               decltype(f_copy)>(out_desc,
                                                 p_out + out_desc.Get1dIndex(n_block_work_begin,
                                                                             k_block_work_begin,
                                                                             ho_block_work_begin,
                                                                             wo_block_work_begin),
                                                 out_block_desc,
                                                 p_out_block,
                                                 f_copy);

        __syncthreads();

        // blockwise convolution
        blockwise_convolution<TFloat,
                              decltype(in_block_desc),
                              decltype(wei_block_desc),
                              decltype(out_block_desc),
                              S,
                              R,
                              InTileSizeH,
                              InTileSizeW,
                              OutTileSizeH,
                              OutTileSizeW,
                              NPerBlock,
                              KPerBlock,
                              YPerBlock,
                              XPerBlock,
                              CPerBlockLoop>(
            in_block_desc, p_in_block, wei_block_desc, p_wei_block, out_block_desc, p_out_block);

        __syncthreads();

        // accum output tensor from LDS to device mem
        blockwise_4d_tensor_op<TFloat,
                               decltype(out_block_desc),
                               decltype(out_desc),
                               1,
                               1,
                               1,
                               64,
                               decltype(f_copy)>(out_block_desc,
                                                 p_out_block,
                                                 out_desc,
                                                 p_out + out_desc.Get1dIndex(n_block_work_begin,
                                                                             k_block_work_begin,
                                                                             ho_block_work_begin,
                                                                             wo_block_work_begin),
                                                 f_copy);
    }
}
