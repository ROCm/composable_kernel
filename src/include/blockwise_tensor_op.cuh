#pragma once
#include "constant_tensor_descriptor.cuh"

#define BLOCKWISE_TENSOR_OP_METHOD 12

#if BLOCKWISE_TENSOR_OP_METHOD == 11
template <class TFloat,
          class SrcDesc,
          class DstDesc,
          unsigned NBlockOpLen0,
          unsigned NBlockOpLen1,
          unsigned NBlockOpLen2,
          unsigned NBlockOpLen3,
          class F,
          unsigned BlockSize>
__device__ void blockwise_4d_tensor_op_binary(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

    static_assert(is_same<decltype(src_desc.GetLengths()), decltype(dst_desc.GetLengths())>::value);

    constexpr auto desc = make_ConstantTensorDescriptor(src_desc.GetLengths());

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(src_desc, "blockwise_4d_tensor_op_binary: src_desc: ");
        print_ConstantTensorDescriptor(dst_desc, "blockwise_4d_tensor_op_binary: dst_desc: ");
    }
#endif

    for(unsigned i = threadIdx.x; i < desc.GetElementSize(); i += BlockSize)
    {
        unsigned is = i;

        const unsigned did0 = is / desc.GetStride(I0);

        is -= did0 * desc.GetStride(I0);

        const unsigned did1 = is / desc.GetStride(I1);

        is -= did1 * desc.GetStride(I1);

        const unsigned did2 = is / desc.GetStride(I2);

        is -= did2 * desc.GetStride(I2);

        const unsigned did3 = is / desc.GetStride(I3);

        const unsigned sindex = src_desc.Get1dIndex(did0, did1, did2, did3);

        const unsigned dindex = dst_desc.Get1dIndex(did0, did1, did2, did3);

        f(p_src[sindex], p_dst[dindex]);
    }
}
#endif

#if BLOCKWISE_TENSOR_OP_METHOD == 12
template <class TFloat,
          class SrcDesc,
          class DstDesc,
          unsigned NBlockOpLen0,
          unsigned NBlockOpLen1,
          unsigned NBlockOpLen2,
          unsigned NBlockOpLen3,
          class F,
          unsigned BlockSize>
__device__ void blockwise_4d_tensor_op_binary(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

    static_assert(is_same<decltype(src_desc.GetLengths()), decltype(dst_desc.GetLengths())>::value);

    constexpr auto desc = make_ConstantTensorDescriptor(src_desc.GetLengths());

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(src_desc, "blockwise_4d_tensor_op_binary: src_desc: ");
        print_ConstantTensorDescriptor(dst_desc, "blockwise_4d_tensor_op_binary: dst_desc: ");
    }
#endif

    constexpr unsigned NLoop = desc.GetElementSize() / BlockSize;

    for(unsigned iloop = 0; iloop < NLoop; ++iloop)
    {
        unsigned is = threadIdx.x + iloop * BlockSize;

        const unsigned did0 = is / desc.GetStride(I0);

        is -= did0 * desc.GetStride(I0);

        const unsigned did1 = is / desc.GetStride(I1);

        is -= did1 * desc.GetStride(I1);

        const unsigned did2 = is / desc.GetStride(I2);

        is -= did2 * desc.GetStride(I2);

        const unsigned did3 = is / desc.GetStride(I3);

        const unsigned sindex = src_desc.Get1dIndex(did0, did1, did2, did3);

        const unsigned dindex = dst_desc.Get1dIndex(did0, did1, did2, did3);

        f(p_src[sindex], p_dst[dindex]);
    }

    constexpr bool has_tail = (desc.GetElementSize() > NLoop * BlockSize);

    if(has_tail)
    {
        unsigned is = threadIdx.x + NLoop * BlockSize;

        if(is < desc.GetElementSize())
        {
            const unsigned did0 = is / desc.GetStride(I0);

            is -= did0 * desc.GetStride(I0);

            const unsigned did1 = is / desc.GetStride(I1);

            is -= did1 * desc.GetStride(I1);

            const unsigned did2 = is / desc.GetStride(I2);

            is -= did2 * desc.GetStride(I2);

            const unsigned did3 = is / desc.GetStride(I3);

            const unsigned sindex = src_desc.Get1dIndex(did0, did1, did2, did3);

            const unsigned dindex = dst_desc.Get1dIndex(did0, did1, did2, did3);

            f(p_src[sindex], p_dst[dindex]);
        }
    }
}

template <class TFloat,
          class DstDesc,
          unsigned NBlockOpLen0,
          unsigned NBlockOpLen1,
          unsigned NBlockOpLen2,
          unsigned NBlockOpLen3,
          class F,
          unsigned BlockSize>
__device__ void blockwise_4d_tensor_op_unary(DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto dst_desc = DstDesc{};

    constexpr auto desc = make_ConstantTensorDescriptor(dst_desc.GetLengths());

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(dst_desc, "blockwise_4d_tensor_op_unary: dst_desc: ");
        print_ConstantTensorDescriptor(desc, "blockwise_4d_tensor_op_unary: desc: ");
    }
#endif

    constexpr unsigned NLoop = desc.GetElementSize() / BlockSize;

    for(unsigned iloop = 0; iloop < NLoop; ++iloop)
    {
        unsigned is = threadIdx.x + iloop * BlockSize;

        const unsigned did0 = is / desc.GetStride(I0);

        is -= did0 * desc.GetStride(I0);

        const unsigned did1 = is / desc.GetStride(I1);

        is -= did1 * desc.GetStride(I1);

        const unsigned did2 = is / desc.GetStride(I2);

        is -= did2 * desc.GetStride(I2);

        const unsigned did3 = is / desc.GetStride(I3);

        const unsigned dindex = dst_desc.Get1dIndex(did0, did1, did2, did3);

        f(p_dst[dindex]);
    }

    constexpr bool has_tail = (desc.GetElementSize() > NLoop * BlockSize);

    if(has_tail)
    {
        unsigned is = threadIdx.x + NLoop * BlockSize;

        if(is < desc.GetElementSize())
        {
            const unsigned did0 = is / desc.GetStride(I0);

            is -= did0 * desc.GetStride(I0);

            const unsigned did1 = is / desc.GetStride(I1);

            is -= did1 * desc.GetStride(I1);

            const unsigned did2 = is / desc.GetStride(I2);

            is -= did2 * desc.GetStride(I2);

            const unsigned did3 = is / desc.GetStride(I3);

            const unsigned dindex = dst_desc.Get1dIndex(did0, did1, did2, did3);

            f(p_dst[dindex]);
        }
    }
}
#endif

#if BLOCKWISE_TENSOR_OP_METHOD == 21
template <class TFloat,
          class SrcDesc,
          class DstDesc,
          unsigned NBlockOpLen0,
          unsigned NBlockOpLen1,
          unsigned NBlockOpLen2,
          unsigned NBlockOpLen3,
          class F,
          unsigned BlockSize>
__device__ void blockwise_4d_tensor_op_binary(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

    static_assert(is_same<decltype(src_desc.GetLengths()), decltype(dst_desc.GetLengths())>::value);

    constexpr unsigned NBlockOpStride3 = 1;
    constexpr unsigned NBlockOpStride2 = NBlockOpLen3 * NBlockOpStride3;
    constexpr unsigned NBlockOpStride1 = NBlockOpLen2 * NBlockOpStride2;
    constexpr unsigned NBlockOpStride0 = NBlockOpLen1 * NBlockOpStride1;

    unsigned itmp = threadIdx.x;

    const unsigned did0_begin = itmp / NBlockOpStride0;

    itmp -= did0_begin * NBlockOpStride0;

    const unsigned did1_begin = itmp / NBlockOpStride1;

    itmp -= did1_begin * NBlockOpStride1;

    const unsigned did2_begin = itmp / NBlockOpStride2;

    itmp -= did2_begin * NBlockOpStride2;

    const unsigned did3_begin = itmp / NBlockOpStride3;

    for(unsigned did0 = did0_begin; did0 < src_desc.GetLength(I0); did0 += NBlockOpLen0)
    {
        for(unsigned did1 = did1_begin; did1 < src_desc.GetLength(I1); did1 += NBlockOpLen1)
        {
            for(unsigned did2 = did2_begin; did2 < src_desc.GetLength(I2); did2 += NBlockOpLen2)
            {
                for(unsigned did3 = did3_begin; did3 < src_desc.GetLength(I3); did3 += NBlockOpLen3)
                {
                    const unsigned sindex = src_desc.Get1dIndex(did0, did1, did2, did3);

                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1, did2, did3);

                    f(p_src[sindex], p_dst[dindex]);
                }
            }
        }
    }
}
#endif

#if BLOCKWISE_TENSOR_OP_METHOD == 22
template <class TFloat,
          class SrcDesc,
          class DstDesc,
          unsigned NBlockOpLen0,
          unsigned NBlockOpLen1,
          unsigned NBlockOpLen2,
          unsigned NBlockOpLen3,
          class F,
          unsigned BlockSize>
__device__ void blockwise_4d_tensor_op_binary(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

    static_assert(is_same<decltype(src_desc.GetLengths()), decltype(dst_desc.GetLengths())>::value);

    constexpr unsigned NBlockOpStride3 = 1;
    constexpr unsigned NBlockOpStride2 = NBlockOpLen3 * NBlockOpStride3;
    constexpr unsigned NBlockOpStride1 = NBlockOpLen2 * NBlockOpStride2;
    constexpr unsigned NBlockOpStride0 = NBlockOpLen1 * NBlockOpStride1;

    unsigned itmp = threadIdx.x;

    const unsigned did0_begin = itmp / NBlockOpStride0;

    itmp -= did0_begin * NBlockOpStride0;

    const unsigned did1_begin = itmp / NBlockOpStride1;

    itmp -= did1_begin * NBlockOpStride1;

    const unsigned did2_begin = itmp / NBlockOpStride2;

    itmp -= did2_begin * NBlockOpStride2;

    const unsigned did3_begin = itmp / NBlockOpStride3;

    unsigned sindex = src_desc.Get1dIndex(did0_begin, did1_begin, did2_begin, did3_begin);
    unsigned dindex = dst_desc.Get1dIndex(did0_begin, did1_begin, did2_begin, did3_begin);

    for(unsigned did0 = did0_begin; did0 < src_desc.GetLength(I0); did0 += NBlockOpLen0)
    {
        const unsigned sindex_save0 = sindex;
        const unsigned dindex_save0 = dindex;

        for(unsigned did1 = did1_begin; did1 < src_desc.GetLength(I1); did1 += NBlockOpLen1)
        {
            const unsigned sindex_save1 = sindex;
            const unsigned dindex_save1 = dindex;

            for(unsigned did2 = did2_begin; did2 < src_desc.GetLength(I2); did2 += NBlockOpLen2)
            {
                const unsigned sindex_save2 = sindex;
                const unsigned dindex_save2 = dindex;

                for(unsigned did3 = did3_begin; did3 < src_desc.GetLength(I3); did3 += NBlockOpLen3)
                {
                    f(p_src[sindex], p_dst[dindex]);

                    sindex += NBlockOpLen3 * src_desc.GetStride(I3);
                    dindex += NBlockOpLen3 * dst_desc.GetStride(I3);
                }

                sindex = sindex_save2 + NBlockOpLen2 * src_desc.GetStride(I2);
                dindex = dindex_save2 + NBlockOpLen2 * dst_desc.GetStride(I2);
            }

            sindex = sindex_save1 + NBlockOpLen1 * src_desc.GetStride(I1);
            dindex = dindex_save1 + NBlockOpLen1 * dst_desc.GetStride(I1);
        }

        sindex = sindex_save0 + NBlockOpLen0 * src_desc.GetStride(I0);
        dindex = dindex_save0 + NBlockOpLen0 * dst_desc.GetStride(I0);
    }
}
#endif

#if BLOCKWISE_TENSOR_OP_METHOD == 23
template <class TFloat,
          class SrcDesc,
          class DstDesc,
          unsigned NBlockOpLen0,
          unsigned NBlockOpLen1,
          unsigned NBlockOpLen2,
          unsigned NBlockOpLen3,
          class F,
          unsigned BlockSize>
__device__ void blockwise_4d_tensor_op_binary(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

    static_assert(is_same<decltype(src_desc.GetLengths()), decltype(dst_desc.GetLengths())>::value);

    constexpr unsigned NBlockOpStride3 = 1;
    constexpr unsigned NBlockOpStride2 = NBlockOpLen3 * NBlockOpStride3;
    constexpr unsigned NBlockOpStride1 = NBlockOpLen2 * NBlockOpStride2;
    constexpr unsigned NBlockOpStride0 = NBlockOpLen1 * NBlockOpStride1;

    unsigned itmp = threadIdx.x;

    const unsigned did0_begin = itmp / NBlockOpStride0;

    itmp -= did0_begin * NBlockOpStride0;

    const unsigned did1_begin = itmp / NBlockOpStride1;

    itmp -= did1_begin * NBlockOpStride1;

    const unsigned did2_begin = itmp / NBlockOpStride2;

    itmp -= did2_begin * NBlockOpStride2;

    const unsigned did3_begin = itmp / NBlockOpStride3;

    unsigned sindex = src_desc.Get1dIndex(did0_begin, did1_begin, did2_begin, did3_begin);
    unsigned dindex = dst_desc.Get1dIndex(did0_begin, did1_begin, did2_begin, did3_begin);

    for(unsigned did0 = did0_begin; did0 < src_desc.GetLength(I0); did0 += NBlockOpLen0)
    {
        unsigned i1 = 0;
        for(unsigned did1 = did1_begin; did1 < src_desc.GetLength(I1); did1 += NBlockOpLen1)
        {
            unsigned i2 = 0;
            for(unsigned did2 = did2_begin; did2 < src_desc.GetLength(I2); did2 += NBlockOpLen2)
            {
                unsigned i3 = 0;
                for(unsigned did3 = did3_begin; did3 < src_desc.GetLength(I3); did3 += NBlockOpLen3)
                {
                    f(p_src[sindex], p_dst[dindex]);

                    sindex += NBlockOpLen3 * src_desc.GetStride(I3);
                    dindex += NBlockOpLen3 * dst_desc.GetStride(I3);

                    ++i3;
                }

                sindex += NBlockOpLen2 * src_desc.GetStride(I2) -
                          i3 * NBlockOpLen3 * src_desc.GetStride(I3);
                dindex += NBlockOpLen2 * dst_desc.GetStride(I2) -
                          i3 * NBlockOpLen3 * dst_desc.GetStride(I3);

                ++i2;
            }

            sindex +=
                NBlockOpLen1 * src_desc.GetStride(I1) - i2 * NBlockOpLen2 * src_desc.GetStride(I2);
            dindex +=
                NBlockOpLen1 * dst_desc.GetStride(I1) - i2 * NBlockOpLen2 * dst_desc.GetStride(I2);

            ++i1;
        }

        sindex +=
            NBlockOpLen0 * src_desc.GetStride(I0) - i1 * NBlockOpLen1 * src_desc.GetStride(I1);
        dindex +=
            NBlockOpLen0 * dst_desc.GetStride(I0) - i1 * NBlockOpLen1 * dst_desc.GetStride(I1);
    }
}
#endif

#if BLOCKWISE_TENSOR_OP_METHOD == 31
template <class TFloat,
          class SrcDesc,
          class DstDesc,
          unsigned NBlockOpLen0,
          unsigned NBlockOpLen1,
          unsigned NBlockOpLen2,
          unsigned NBlockOpLen3,
          class F,
          unsigned BlockSize>
__device__ void blockwise_4d_tensor_op_binary(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

    static_assert(is_same<decltype(src_desc.GetLengths()), decltype(dst_desc.GetLengths())>::value);

    constexpr unsigned NBlockOpStride3 = 1;
    constexpr unsigned NBlockOpStride2 = NBlockOpLen3 * NBlockOpStride3;
    constexpr unsigned NBlockOpStride1 = NBlockOpLen2 * NBlockOpStride2;
    constexpr unsigned NBlockOpStride0 = NBlockOpLen1 * NBlockOpStride1;

    unsigned itmp = threadIdx.x;

    const unsigned did0_begin = itmp / NBlockOpStride0;

    itmp -= did0_begin * NBlockOpStride0;

    const unsigned did1_begin = itmp / NBlockOpStride1;

    itmp -= did1_begin * NBlockOpStride1;

    const unsigned did2_begin = itmp / NBlockOpStride2;

    itmp -= did2_begin * NBlockOpStride2;

    const unsigned did3_begin = itmp / NBlockOpStride3;

    for(unsigned did0 = did0_begin; did0 < src_desc.GetLength(I0); did0 += NBlockOpLen0)
    {
        for(unsigned did1 = did1_begin; did1 < src_desc.GetLength(I1); did1 += NBlockOpLen1)
        {
            for(unsigned did2 = did2_begin; did2 < src_desc.GetLength(I2); did2 += NBlockOpLen2)
            {
                for(unsigned did3 = did3_begin; did3 < src_desc.GetLength(I3); did3 += NBlockOpLen3)
                {
                    const unsigned sindex = src_desc.Get1dIndex(did0, did1, did2, did3);

                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1, did2, did3);

                    f(p_src[sindex], p_dst[dindex]);
                }
            }
        }
    }
}
#endif
