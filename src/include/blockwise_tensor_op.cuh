#pragma once
#include "constant_tensor_descriptor.cuh"

#if 0
template <class TFloat,
          class SrcDesc,
          class DstDesc,
          unsigned NWorkLen0,
          unsigned NWorkLen1,
          unsigned NWorkLen2,
          unsigned NWorkLen3,
          class F,
          unsigned BlockSize>
__device__ void blockwise_4d_tensor_op(
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
        print_ConstantTensorDescriptor(src_desc, "blockwise_4d_tensor_op: src_desc: ");
        print_ConstantTensorDescriptor(dst_desc, "blockwise_4d_tensor_op: dst_desc: ");
    }
#endif

    constexpr unsigned NWorkStride3 = 1;
    constexpr unsigned NWorkStride2 = NWorkLen3 * NWorkStride3;
    constexpr unsigned NWorkStride1 = NWorkLen2 * NWorkStride2;
    constexpr unsigned NWorkStride0 = NWorkLen1 * NWorkStride1;

    unsigned itmp =
        threadIdx.x;

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

#elif 1

template <class TFloat,
          class SrcDesc,
          class DstDesc,
          unsigned NWorkLen0,
          unsigned NWorkLen1,
          unsigned NWorkLen2,
          unsigned NWorkLen3,
          class F,
          unsigned BlockSize>
__device__ void blockwise_4d_tensor_op(
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
        print_ConstantTensorDescriptor(src_desc, "blockwise_4d_tensor_op: src_desc: ");
        print_ConstantTensorDescriptor(dst_desc, "blockwise_4d_tensor_op: dst_desc: ");
    }
#endif

    unsigned lid = threadIdx.x;

    for(unsigned i = lid; i < src_desc.GetElementSize(); i += BlockSize)
    {
        unsigned is = i;

        const unsigned did0 = is / src_desc.GetStride(I0);

        is -= did0 * src_desc.GetStride(I0);

        const unsigned did1 = is / src_desc.GetStride(I1);

        is -= did1 * src_desc.GetStride(I1);

        const unsigned did2 = is / src_desc.GetStride(I2);

        is -= did2 * src_desc.GetStride(I2);

        const unsigned did3 = is / src_desc.GetStride(I3);

        const unsigned sindex = src_desc.GetStride(I0) * did0 + src_desc.GetStride(I1) * did1 +
                                src_desc.GetStride(I2) * did2 + src_desc.GetStride(I3) * did3;

        const unsigned dindex = dst_desc.GetStride(I0) * did0 + dst_desc.GetStride(I1) * did1 +
                                dst_desc.GetStride(I2) * did2 + dst_desc.GetStride(I3) * did3;

        f(p_src[sindex], p_dst[dindex]);
    }
}
#endif
