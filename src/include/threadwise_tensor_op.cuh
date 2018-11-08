#pragma once
#include "constant_tensor_descriptor.cuh"

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

    static_assert(is_same<decltype(src_desc.GetLengths()), decltype(dst_desc.GetLengths())>::value);

#if 0
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
