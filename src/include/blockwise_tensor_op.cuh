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

template <class TFloat, class DescA, class DescB, class DescRef, class F, unsigned BlockSize>
__device__ void blockwise_4d_tensor_pointwise_op_binary(
    DescA, TFloat* const __restrict__ p_a, DescB, TFloat* __restrict__ p_b, DescRef, F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto desc_a   = DescA{};
    constexpr auto desc_b   = DescB{};
    constexpr auto desc_ref = DescRef{};

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(desc_a, "blockwise_4d_tensor_op_binary: desc_a: ");
        print_ConstantTensorDescriptor(desc_b, "blockwise_4d_tensor_op_binary: desc_b: ");
        print_ConstantTensorDescriptor(desc_ref, "blockwise_4d_tensor_op_binary: desc_ref: ");
    }
#endif

    constexpr unsigned NLoop = desc_ref.GetElementSize() / BlockSize;

    for(unsigned iloop = 0; iloop < NLoop; ++iloop)
    {
        unsigned is = threadIdx.x + iloop * BlockSize;

        const unsigned did0 = is / desc_ref.GetStride(I0);

        is -= did0 * desc_ref.GetStride(I0);

        const unsigned did1 = is / desc_ref.GetStride(I1);

        is -= did1 * desc_ref.GetStride(I1);

        const unsigned did2 = is / desc_ref.GetStride(I2);

        is -= did2 * desc_ref.GetStride(I2);

        const unsigned did3 = is / desc_ref.GetStride(I3);

        const unsigned aindex = desc_a.Get1dIndex(did0, did1, did2, did3);

        const unsigned bindex = desc_b.Get1dIndex(did0, did1, did2, did3);

        f(p_a[aindex], p_b[bindex]);
    }

    constexpr bool has_tail = (desc_ref.GetElementSize() > NLoop * BlockSize);

    if(has_tail)
    {
        unsigned is = threadIdx.x + NLoop * BlockSize;

        if(is < desc_ref.GetElementSize())
        {
            const unsigned did0 = is / desc_ref.GetStride(I0);

            is -= did0 * desc_ref.GetStride(I0);

            const unsigned did1 = is / desc_ref.GetStride(I1);

            is -= did1 * desc_ref.GetStride(I1);

            const unsigned did2 = is / desc_ref.GetStride(I2);

            is -= did2 * desc_ref.GetStride(I2);

            const unsigned did3 = is / desc_ref.GetStride(I3);

            const unsigned aindex = desc_a.Get1dIndex(did0, did1, did2, did3);

            const unsigned bindex = desc_b.Get1dIndex(did0, did1, did2, did3);

            f(p_a[aindex], p_b[bindex]);
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

template <class TFloat, class SrcDesc, class DstDesc, class RefDesc, unsigned BlockSize>
__device__ void blockwise_4d_tensor_copy(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, RefDesc)
{
    auto f_copy = [](const TFloat& src, TFloat& dst) { dst = src; };

    blockwise_4d_tensor_pointwise_op_binary<TFloat,
                                            SrcDesc,
                                            DstDesc,
                                            RefDesc,
                                            decltype(f_copy),
                                            BlockSize>(
        SrcDesc{}, p_src, DstDesc{}, p_dst, RefDesc{}, f_copy);
}
