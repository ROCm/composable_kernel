#pragma once
#include "constant_tensor_descriptor.cuh"

template <class TFloat, class DstDesc, class F, unsigned BlockSize>
__device__ void blockwise_4d_tensor_pointwise_op_unary(DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

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

template <class TFloat, class SrcDesc, class DstDesc, class F, unsigned BlockSize>
__device__ void blockwise_4d_tensor_pointwise_op_binary(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

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

template <class TFloat, class DstDesc, unsigned BlockSize>
__device__ void blockwise_4d_tensor_set_zero(DstDesc, TFloat* __restrict__ p_dst)
{
    auto f_set_zero = [](TFloat& v) { v = TFloat(0); };

    blockwise_4d_tensor_pointwise_op_unary<TFloat, DstDesc, decltype(f_set_zero), BlockSize>(
        DstDesc{}, p_dst, f_set_zero);
}

template <class TFloat, class SrcDesc, class DstDesc, unsigned BlockSize>
__device__ void blockwise_4d_tensor_copy(SrcDesc,
                                         TFloat* const __restrict__ p_src,
                                         DstDesc,
                                         TFloat* __restrict__ p_dst)
{
    auto f_copy = [](const TFloat& src, TFloat& dst) { dst = src; };

    blockwise_4d_tensor_pointwise_op_binary<TFloat, SrcDesc, DstDesc, decltype(f_copy), BlockSize>(
        SrcDesc{}, p_src, DstDesc{}, p_dst, f_copy);
}

template <class TFloat, class SrcDesc, class DstDesc, unsigned BlockSize>
__device__ void blockwise_4d_tensor_accumulate(SrcDesc,
                                               TFloat* const __restrict__ p_src,
                                               DstDesc,
                                               TFloat* __restrict__ p_dst)
{
    auto f_accum = [](const TFloat& src, TFloat& dst) { dst += src; };

    blockwise_4d_tensor_pointwise_op_binary<TFloat, SrcDesc, DstDesc, decltype(f_accum), BlockSize>(
        SrcDesc{}, p_src, DstDesc{}, p_dst, f_accum);
}