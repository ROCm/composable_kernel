#pragma once
#include "constant_tensor_descriptor.cuh"

template <class TFloat, class Desc, class F>
__device__ void threadwise_4d_tensor_pointwise_op_unary(Desc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto desc = Desc{};

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(desc, "threadwise_4d_tensor_op_unary: ");
    }
#endif

    for(unsigned did0 = 0; did0 < desc.GetLength(I0); ++did0)
    {
        for(unsigned did1 = 0; did1 < desc.GetLength(I1); ++did1)
        {
            for(unsigned did2 = 0; did2 < desc.GetLength(I2); ++did2)
            {
                for(unsigned did3 = 0; did3 < desc.GetLength(I3); ++did3)
                {
                    const unsigned dindex = desc.Get1dIndex(did0, did1, did2, did3);

                    f(p_dst[dindex]);
                }
            }
        }
    }
}

template <class TFloat, class SrcDesc, class DstDesc, class F>
__device__ void threadwise_4d_tensor_pointwise_op_binary(
    SrcDesc, TFloat* const __restrict__ p_src, DstDesc, TFloat* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};

    static_assert(is_same<decltype(src_desc.GetLengths()), decltype(dst_desc.GetLengths())>::value);

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(src_desc, "threadwise_4d_tensor_op_binary: src_desc: ");
        print_ConstantTensorDescriptor(dst_desc, "threadwise_4d_tensor_op_binary: dst_desc: ");
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
                    const unsigned sindex = src_desc.Get1dIndex(did0, did1, did2, did3);

                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1, did2, did3);

                    f(p_src[sindex], p_dst[dindex]);
                }
            }
        }
    }
}

template <class TFloat, class Desc>
__device__ void threadwise_4d_tensor_set_zero(Desc, TFloat* __restrict__ p_dst)
{
    auto f_set_zero = [](TFloat& v) { v = TFloat(0); };

    threadwise_4d_tensor_pointwise_op_unary<TFloat, Desc, decltype(f_set_zero)>(
        Desc{}, p_dst, f_set_zero);
}

template <class TFloat, class SrcDesc, class DstDesc>
__device__ void threadwise_4d_tensor_copy(SrcDesc,
                                          TFloat* const __restrict__ p_src,
                                          DstDesc,
                                          TFloat* __restrict__ p_dst)
{
    auto f_copy = [](const TFloat& src, TFloat& dst) { dst = src; };

    threadwise_4d_tensor_pointwise_op_binary<TFloat, SrcDesc, DstDesc, decltype(f_copy)>(
        SrcDesc{}, p_src, DstDesc{}, p_dst, f_copy);
}