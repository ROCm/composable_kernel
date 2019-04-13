#pragma once
#include "ConstantTensorDescriptor.hip.hpp"

template <class Float, class Desc, class F>
__device__ void threadwise_2d_tensor_pointwise_operation_unary(Desc, Float* __restrict__ p, F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    constexpr auto desc = Desc{};

#if 0
    if(get_thread_local_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(desc, "threadwise_4d_tensor_op_unary: ");
    }
#endif

    for(index_t did0 = 0; did0 < desc.GetLength(I0); ++did0)
    {
        for(index_t did1 = 0; did1 < desc.GetLength(I1); ++did1)
        {
            const index_t dindex = desc.Get1dIndex(did0, did1);

            f(p[dindex]);
        }
    }
}

// TODO: in order to optimize mem access for different mem type,
// need to write specialized version
template <class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class DstFromSrcReorder,
          class F>
__device__ void threadwise_2d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src(
    SrcDesc,
    Float* const __restrict__ p_src,
    DstDesc,
    Float* __restrict__ p_dst,
    SrcOpLengths,
    DstFromSrcReorder,
    F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    constexpr index_t IR0 = DstFromSrcReorder{}.Get(I0);
    constexpr index_t IR1 = DstFromSrcReorder{}.Get(I1);

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    for(index_t did0 = 0; did0 < ref_desc.GetLength(I0); ++did0)
    {
        for(index_t did1 = 0; did1 < ref_desc.GetLength(I1); ++did1)
        {
            const index_t aindex = src_desc.Get1dIndex(did0, did1);

            const index_t did[2] = {did0, did1};

            const index_t bindex = dst_desc.Get1dIndex(did[IR0], did[IR1]);

            f(p_src[aindex], p_dst[bindex]);
        }
    }
}

template <class Float, class Desc>
__device__ void threadwise_2d_tensor_set_zero(Desc, Float* __restrict__ p)
{
    auto f_set_zero = [](Float& v) { v = Float(0); };

    threadwise_2d_tensor_pointwise_operation_unary<Float, Desc, decltype(f_set_zero)>(
        Desc{}, p, f_set_zero);
}

template <class Float, class SrcDesc, class DstDesc, class SrcOpLengths, class DstFromSrcReorder>
__device__ void
threadwise_2d_tensor_copy_reorder_by_get_dst_from_src(SrcDesc,
                                                      Float* const __restrict__ p_src,
                                                      DstDesc,
                                                      Float* __restrict__ p_dst,
                                                      SrcOpLengths,
                                                      DstFromSrcReorder)
{
    auto f_copy = [](const Float& src, Float& dst) { dst = src; };

    threadwise_2d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src(
        SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, DstFromSrcReorder{}, f_copy);
}

template <class Float, class SrcDesc, class DstDesc, class SrcOpLengths>
__device__ void threadwise_2d_tensor_copy(
    SrcDesc, Float* const __restrict__ p_src, DstDesc, Float* __restrict__ p_dst, SrcOpLengths)
{
    auto dst_from_src_reorder = Sequence<0, 1>{};

    threadwise_2d_tensor_copy_reorder_by_get_dst_from_src(
        SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, dst_from_src_reorder);
}

template <class Float, class Desc, class IDim, class NShift>
__device__ void threadwise_2d_tensor_shift_down(Desc, Float* __restrict__ p, IDim, NShift)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    constexpr auto desc = Desc{};

#if 0
    if(get_thread_local_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(desc, "threadwise_4d_tensor_shift_down: ");
    }
#endif

    constexpr index_t nshift = NShift::mValue;

    constexpr index_t did0_end =
        is_same<decltype(I0), IDim>::value ? desc.GetLength(I0) - nshift : desc.GetLength(I0);

    constexpr index_t did1_end =
        is_same<decltype(I1), IDim>::value ? desc.GetLength(I1) - nshift : desc.GetLength(I1);

    for(index_t did0 = 0; did0 < did0_end; ++did0)
    {
        for(index_t did1 = 0; did1 < did1_end; ++did1)
        {
            const index_t dindex = desc.Get1dIndex(did0, did1);

            const index_t sindex = dindex + nshift * desc.GetStride(IDim{});

            p[dindex] = p[sindex];
        }
    }
}
