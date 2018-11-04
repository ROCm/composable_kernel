#pragma once
#include "device_tensor_descriptor.cuh"

template <class TFloat,
          unsigned NWorkLen0,
          unsigned NWorkLen1,
          unsigned NWorkLen2,
          unsigned NWorkLen3,
          class F>
__device__ void blockwise_4d_tensor_op(const DeviceTensorDescriptor<4>& src_desc,
                                       TFloat* const __restrict__ p_src,
                                       const DeviceTensorDescriptor<4>& dst_desc,
                                       TFloat* __restrict__ p_dst,
                                       F f)
{
#if 0
    if(threadIdx.x == 0)
    {
        printf("blockwise_4d_tensor_op: 0: \t"
               "threadIdx.x %u \t"
               "src_desc {%u %u %u %u}, {%u %u %u %u}\t"
               "dst_desc {%u %u %u %u}, {%u %u %u %u}\n",
               threadIdx.x,
               src_desc.GetLength(0),
               src_desc.GetLength(1),
               src_desc.GetLength(2),
               src_desc.GetLength(3),
               src_desc.GetStride(0),
               src_desc.GetStride(1),
               src_desc.GetStride(2),
               src_desc.GetStride(3),
               dst_desc.GetLength(0),
               dst_desc.GetLength(1),
               dst_desc.GetLength(2),
               dst_desc.GetLength(3),
               dst_desc.GetStride(0),
               dst_desc.GetStride(1),
               dst_desc.GetStride(2),
               dst_desc.GetStride(3));
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

    for(unsigned did0 = did0_begin; did0 < src_desc.GetLength(0); did0 += NWorkLen0)
    {
        for(unsigned did1 = did1_begin; did1 < src_desc.GetLength(1); did1 += NWorkLen1)
        {
            for(unsigned did2 = did2_begin; did2 < src_desc.GetLength(2); did2 += NWorkLen2)
            {
                for(unsigned did3 = did3_begin; did3 < src_desc.GetLength(3); did3 += NWorkLen3)
                {
                    const unsigned sindex =
                        src_desc.GetStride(0) * did0 + src_desc.GetStride(1) * did1 +
                        src_desc.GetStride(2) * did2 + src_desc.GetStride(3) * did3;

                    const unsigned dindex =
                        dst_desc.GetStride(0) * did0 + dst_desc.GetStride(1) * did1 +
                        dst_desc.GetStride(2) * did2 + dst_desc.GetStride(3) * did3;

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

template <class TFloat, class F>
__device__ void threadwise_4d_tensor_op(const DeviceTensorDescriptor<4>& src_desc,
                                        TFloat* const __restrict__ p_src,
                                        const DeviceTensorDescriptor<4>& dst_desc,
                                        TFloat* __restrict__ p_dst,
                                        F f)
{
#if 0
    if(threadIdx.x == 0)
    {
        printf("threadwise_4d_tensor_op: 0: \t"
               "threadIdx.x %u \t"
               "src_desc {%u %u %u %u}, {%u %u %u %u}\t"
               "dst_desc {%u %u %u %u}, {%u %u %u %u}\n",
               threadIdx.x,
               src_desc.GetLength(0),
               src_desc.GetLength(1),
               src_desc.GetLength(2),
               src_desc.GetLength(3),
               src_desc.GetStride(0),
               src_desc.GetStride(1),
               src_desc.GetStride(2),
               src_desc.GetStride(3),
               dst_desc.GetLength(0),
               dst_desc.GetLength(1),
               dst_desc.GetLength(2),
               dst_desc.GetLength(3),
               dst_desc.GetStride(0),
               dst_desc.GetStride(1),
               dst_desc.GetStride(2),
               dst_desc.GetStride(3));
    }
#endif

    for(unsigned did0 = 0; did0 < src_desc.GetLength(0); ++did0)
    {
        for(unsigned did1 = 0; did1 < src_desc.GetLength(1); ++did1)
        {
            for(unsigned did2 = 0; did2 < src_desc.GetLength(2); ++did2)
            {
                for(unsigned did3 = 0; did3 < src_desc.GetLength(3); ++did3)
                {
                    const unsigned sindex =
                        src_desc.GetStride(0) * did0 + src_desc.GetStride(1) * did1 +
                        src_desc.GetStride(2) * did2 + src_desc.GetStride(3) * did3;

                    const unsigned dindex =
                        dst_desc.GetStride(0) * did0 + dst_desc.GetStride(1) * did1 +
                        dst_desc.GetStride(2) * did2 + dst_desc.GetStride(3) * did3;

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

template <class TFloat>
__device__ void threadwise_direct_convolution(const DeviceTensorDescriptor<4>& in_desc,
                                              TFloat* const __restrict__ p_in,
                                              const DeviceTensorDescriptor<4>& wei_desc,
                                              TFloat* const __restrict__ p_wei,
                                              const DeviceTensorDescriptor<4>& out_desc,
                                              TFloat* __restrict__ p_out)
{
#if 0
    if(threadIdx.x == 0)
    {
        printf("threadwise_direct_convolution: 0: \t"
               "threadIdx.x %u \t"
               "in_desc {%u %u %u %u}, {%u %u %u %u}\t"
               "wei_desc {%u %u %u %u}, {%u %u %u %u}\t"
               "out_desc {%u %u %u %u}, {%u %u %u %u}\n",
               threadIdx.x,
               in_desc.GetLength(0),
               in_desc.GetLength(1),
               in_desc.GetLength(2),
               in_desc.GetLength(3),
               in_desc.GetStride(0),
               in_desc.GetStride(1),
               in_desc.GetStride(2),
               in_desc.GetStride(3),
               wei_desc.GetLength(0),
               wei_desc.GetLength(1),
               wei_desc.GetLength(2),
               wei_desc.GetLength(3),
               wei_desc.GetStride(0),
               wei_desc.GetStride(1),
               wei_desc.GetStride(2),
               wei_desc.GetStride(3),
               out_desc.GetLength(0),
               out_desc.GetLength(1),
               out_desc.GetLength(2),
               out_desc.GetLength(3),
               out_desc.GetStride(0),
               out_desc.GetStride(1),
               out_desc.GetStride(2),
               out_desc.GetStride(3));
    }
#elif 0
    {
        printf("threadwise_direct_convolution: 0: \t"
               "threadIdx.x %u \t"
               "p_in %f %f %f %f %f %f %f %f, \t"
               "p_wei %f %f %f %f %f %f %f %f %f, \t"
               "p_out %f %f %f %f, \n",
               threadIdx.x,
               p_in[0],
               p_in[1],
               p_in[2],
               p_in[3],
               p_in[4],
               p_in[5],
               p_in[6],
               p_in[7],
               p_wei[0],
               p_wei[1],
               p_wei[2],
               p_wei[3],
               p_wei[4],
               p_wei[5],
               p_wei[6],
               p_wei[7],
               p_wei[8],
               p_out[0],
               p_out[1],
               p_out[2],
               p_out[3]);
    }
#endif

    for(unsigned n = 0; n < out_desc.GetLength(0); ++n)
    {
        for(unsigned k = 0; k < out_desc.GetLength(1); ++k)
        {
            for(unsigned ho = 0; ho < out_desc.GetLength(2); ++ho)
            {
                for(unsigned wo = 0; wo < out_desc.GetLength(3); ++wo)
                {
                    for(unsigned c = 0; c < wei_desc.GetLength(1); ++c)
                    {
                        for(unsigned s = 0; s < wei_desc.GetLength(2); ++s)
                        {
                            for(unsigned r = 0; r < wei_desc.GetLength(3); ++r)
                            {
                                const unsigned hi = ho + s;
                                const unsigned wi = wo + r;

                                const unsigned in_index =
                                    in_desc.GetStride(0) * n + in_desc.GetStride(1) * c +
                                    in_desc.GetStride(2) * hi + in_desc.GetStride(3) * wi;

                                const unsigned wei_index =
                                    wei_desc.GetStride(0) * k + wei_desc.GetStride(1) * c +
                                    wei_desc.GetStride(2) * s + in_desc.GetStride(3) * r;

                                const unsigned out_index =
                                    out_desc.GetStride(0) * n + out_desc.GetStride(1) * k +
                                    out_desc.GetStride(2) * ho + out_desc.GetStride(3) * wo;

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
          unsigned S, // weight size in H-dir
          unsigned R, // weight size in W-dir
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned NPerBlock,
          unsigned KPerBlock,
          unsigned YPerBlock,
          unsigned XPerBlock,
          unsigned CPerBlockLoop>
__device__ void blockwise_convolution(const DeviceTensorDescriptor<4>& in_desc,
                                      TFloat* const __restrict__ p_in,
                                      const DeviceTensorDescriptor<4>& wei_desc,
                                      TFloat* const __restrict__ p_wei,
                                      const DeviceTensorDescriptor<4>& out_desc,
                                      TFloat* __restrict__ p_out)
{
#if 0
    if(threadIdx.x == 0)
    {
        printf("blockwise_convolution: 0: \t"
               "threadIdx.x %u \t"
               "in_desc {%u %u %u %u}, {%u %u %u %u}\t"
               "wei_desc {%u %u %u %u}, {%u %u %u %u}\t"
               "out_desc {%u %u %u %u}, {%u %u %u %u}\n",
               threadIdx.x,
               in_desc.GetLength(0),
               in_desc.GetLength(1),
               in_desc.GetLength(2),
               in_desc.GetLength(3),
               in_desc.GetStride(0),
               in_desc.GetStride(1),
               in_desc.GetStride(2),
               in_desc.GetStride(3),
               wei_desc.GetLength(0),
               wei_desc.GetLength(1),
               wei_desc.GetLength(2),
               wei_desc.GetLength(3),
               wei_desc.GetStride(0),
               wei_desc.GetStride(1),
               wei_desc.GetStride(2),
               wei_desc.GetStride(3),
               out_desc.GetLength(0),
               out_desc.GetLength(1),
               out_desc.GetLength(2),
               out_desc.GetLength(3),
               out_desc.GetStride(0),
               out_desc.GetStride(1),
               out_desc.GetStride(2),
               out_desc.GetStride(3));
    }
#endif

    // for now, one thread do 1 N and 1 K
    DeviceTensorDescriptor<4> in_thread_src_desc = in_desc;
    in_thread_src_desc.mpLengths[0]              = 1;
    in_thread_src_desc.mpLengths[1]              = CPerBlockLoop;
    in_thread_src_desc.mpLengths[2]              = OutTileSizeH + S - 1;
    in_thread_src_desc.mpLengths[3]              = OutTileSizeW + R - 1;

    DeviceTensorDescriptor<4> wei_thread_src_desc = wei_desc;
    wei_thread_src_desc.mpLengths[0]              = 1;
    wei_thread_src_desc.mpLengths[1]              = CPerBlockLoop;
    wei_thread_src_desc.mpLengths[2]              = S;
    wei_thread_src_desc.mpLengths[3]              = R;

    DeviceTensorDescriptor<4> out_thread_src_desc = out_desc;
    out_thread_src_desc.mpLengths[0]              = 1;
    out_thread_src_desc.mpLengths[1]              = 1;
    out_thread_src_desc.mpLengths[2]              = OutTileSizeH;
    out_thread_src_desc.mpLengths[3]              = OutTileSizeW;

    DeviceTensorDescriptor<4> in_thread_dst_desc = in_thread_src_desc;
    in_thread_dst_desc.mpStrides[3]              = 1;
    in_thread_dst_desc.mpStrides[2] =
        in_thread_dst_desc.GetLength(3) * in_thread_dst_desc.GetStride(3);
    in_thread_dst_desc.mpStrides[1] =
        in_thread_dst_desc.GetLength(2) * in_thread_dst_desc.GetStride(2);
    in_thread_dst_desc.mpStrides[0] =
        in_thread_dst_desc.GetLength(1) * in_thread_dst_desc.GetStride(1);

    DeviceTensorDescriptor<4> wei_thread_dst_desc = wei_thread_src_desc;
    wei_thread_dst_desc.mpStrides[3]              = 1;
    wei_thread_dst_desc.mpStrides[2] =
        wei_thread_dst_desc.GetLength(3) * wei_thread_dst_desc.GetStride(3);
    wei_thread_dst_desc.mpStrides[1] =
        wei_thread_dst_desc.GetLength(2) * wei_thread_dst_desc.GetStride(2);
    wei_thread_dst_desc.mpStrides[0] =
        wei_thread_dst_desc.GetLength(1) * wei_thread_dst_desc.GetStride(1);

    DeviceTensorDescriptor<4> out_thread_dst_desc = out_thread_src_desc;
    out_thread_dst_desc.mpStrides[3]              = 1;
    out_thread_dst_desc.mpStrides[2] =
        out_thread_dst_desc.GetLength(3) * out_thread_dst_desc.GetStride(3);
    out_thread_dst_desc.mpStrides[1] =
        out_thread_dst_desc.GetLength(2) * out_thread_dst_desc.GetStride(2);
    out_thread_dst_desc.mpStrides[0] =
        out_thread_dst_desc.GetLength(1) * out_thread_dst_desc.GetStride(1);

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
        threadwise_4d_tensor_op<TFloat, decltype(f_copy)>(
            in_thread_src_desc,
            p_in + in_desc.Get1dIndex(
                       n_thread_work_begin, 0, hi_thread_work_begin, wi_thread_work_begin),
            in_thread_dst_desc,
            p_in_thread,
            f_copy);

        // copy weight tensor into register
        threadwise_4d_tensor_op<TFloat, decltype(f_copy)>(
            wei_thread_src_desc,
            p_wei + wei_desc.Get1dIndex(k_thread_work_begin, 0, 0, 0),
            wei_thread_dst_desc,
            p_wei_thread,
            f_copy);

        // copy output tensor into register
        threadwise_4d_tensor_op<TFloat, decltype(f_copy)>(
            out_thread_src_desc,
            p_out + out_desc.Get1dIndex(n_thread_work_begin,
                                        k_thread_work_begin,
                                        ho_thread_work_begin,
                                        wo_thread_work_begin),
            out_thread_dst_desc,
            p_out_thread,
            f_copy);

        // threadwise convolution
        threadwise_direct_convolution(in_thread_dst_desc,
                                      p_in_thread,
                                      wei_thread_dst_desc,
                                      p_wei_thread,
                                      out_thread_dst_desc,
                                      p_out_thread);

        // accumulate output tensor into device mem
        threadwise_4d_tensor_op<TFloat, decltype(f_copy)>(
            out_thread_dst_desc,
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
          unsigned S, // weight size in H-dir
          unsigned R, // weight size in W-dir
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned NPerBlock,
          unsigned KPerBlock,
          unsigned YPerBlock,
          unsigned XPerBlock,
          unsigned CPerBlockLoop>
__global__ void gridwise_convolution(const DeviceTensorDescriptor<4> in_desc,
                                     TFloat* const __restrict__ p_in,
                                     const DeviceTensorDescriptor<4> wei_desc,
                                     TFloat* const __restrict__ p_wei,
                                     const DeviceTensorDescriptor<4> out_desc,
                                     TFloat* __restrict__ p_out)
{
#if 0
    if(threadIdx.x == 0)
    {
        printf("gridwise_convolution: 0: \t"
               "threadIdx.x %u \t"
               "in_desc {%u %u %u %u}, {%u %u %u %u}\t"
               "wei_desc {%u %u %u %u}, {%u %u %u %u}\t"
               "out_desc {%u %u %u %u}, {%u %u %u %u}\n",
               threadIdx.x,
               in_desc.GetLength(0),
               in_desc.GetLength(1),
               in_desc.GetLength(2),
               in_desc.GetLength(3),
               in_desc.GetStride(0),
               in_desc.GetStride(1),
               in_desc.GetStride(2),
               in_desc.GetStride(3),
               wei_desc.GetLength(0),
               wei_desc.GetLength(1),
               wei_desc.GetLength(2),
               wei_desc.GetLength(3),
               wei_desc.GetStride(0),
               wei_desc.GetStride(1),
               wei_desc.GetStride(2),
               wei_desc.GetStride(3),
               out_desc.GetLength(0),
               out_desc.GetLength(1),
               out_desc.GetLength(2),
               out_desc.GetLength(3),
               out_desc.GetStride(0),
               out_desc.GetStride(1),
               out_desc.GetStride(2),
               out_desc.GetStride(3));
    }
#endif

    const unsigned NBlockWork = (in_desc.GetLength(0) + NPerBlock - 1) / NPerBlock;
    const unsigned YBlockWork = (in_desc.GetLength(2) + YPerBlock - 1) / YPerBlock;
    const unsigned XBlockWork = (in_desc.GetLength(3) + XPerBlock - 1) / XPerBlock;

    const unsigned KBlockWork = (wei_desc.GetLength(1) + KPerBlock - 1) / KPerBlock;

    const unsigned block_id =
        blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.y * gridDim.x);

    // this is ugly
    DeviceTensorDescriptor<4> wei_block_desc;
    wei_block_desc.mpLengths[0] = KPerBlock;
    wei_block_desc.mpLengths[1] = CPerBlockLoop;
    wei_block_desc.mpLengths[2] = S;
    wei_block_desc.mpLengths[3] = R;
    wei_block_desc.mpStrides[3] = 1;
    wei_block_desc.mpStrides[2] = wei_block_desc.GetLength(3) * wei_block_desc.GetStride(3);
    wei_block_desc.mpStrides[1] = wei_block_desc.GetLength(2) * wei_block_desc.GetStride(2);
    wei_block_desc.mpStrides[0] = wei_block_desc.GetLength(1) * wei_block_desc.GetStride(1);

    DeviceTensorDescriptor<4> out_block_desc;
    out_block_desc.mpLengths[0] = NPerBlock;
    out_block_desc.mpLengths[1] = KPerBlock;
    out_block_desc.mpLengths[2] = YPerBlock * OutTileSizeH;
    out_block_desc.mpLengths[3] = XPerBlock * OutTileSizeW;
    out_block_desc.mpStrides[3] = 1;
    out_block_desc.mpStrides[2] = out_block_desc.GetLength(3) * out_block_desc.GetStride(3);
    out_block_desc.mpStrides[1] = out_block_desc.GetLength(2) * out_block_desc.GetStride(2);
    out_block_desc.mpStrides[0] = out_block_desc.GetLength(1) * out_block_desc.GetStride(1);

    DeviceTensorDescriptor<4> in_block_desc;
    in_block_desc.mpLengths[0] = NPerBlock;
    in_block_desc.mpLengths[1] = CPerBlockLoop;
    in_block_desc.mpLengths[2] = YPerBlock * OutTileSizeH + S - 1;
    in_block_desc.mpLengths[3] = XPerBlock * OutTileSizeW + R - 1;
    in_block_desc.mpStrides[3] = 1;
    in_block_desc.mpStrides[2] = in_block_desc.GetLength(3) * in_block_desc.GetStride(3);
    in_block_desc.mpStrides[1] = in_block_desc.GetLength(2) * in_block_desc.GetStride(2);
    in_block_desc.mpStrides[0] = in_block_desc.GetLength(1) * in_block_desc.GetStride(1);

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

    for(unsigned c_block_work_begin = 0; c_block_work_begin < in_desc.GetLength(1);
        c_block_work_begin += CPerBlockLoop)
    {
        auto f_copy = [](const TFloat& src, TFloat& dst) { dst = src; };

        // copy input tensor to LDS
        blockwise_4d_tensor_op<TFloat, 1, 1, 1, 64, decltype(f_copy)>(
            in_desc,
            p_in + in_desc.Get1dIndex(n_block_work_begin,
                                      c_block_work_begin,
                                      hi_block_work_begin,
                                      wi_block_work_begin),
            in_block_desc,
            p_in_block,
            f_copy);

        // copy weight tensor to LDS
        blockwise_4d_tensor_op<TFloat, 1, 1, 1, 64, decltype(f_copy)>(
            wei_desc,
            p_wei + wei_desc.Get1dIndex(k_block_work_begin, c_block_work_begin, 0, 0),
            wei_block_desc,
            p_wei_block,
            f_copy);

        // copy output tensor to LDS
        blockwise_4d_tensor_op<TFloat, 1, 1, 1, 64, decltype(f_copy)>(
            out_desc,
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
        blockwise_4d_tensor_op<TFloat, 1, 1, 1, 64, decltype(f_copy)>(
            out_block_desc,
            p_out_block,
            out_desc,
            p_out + out_desc.Get1dIndex(n_block_work_begin,
                                        k_block_work_begin,
                                        ho_block_work_begin,
                                        wo_block_work_begin),
            f_copy);
    }
}
