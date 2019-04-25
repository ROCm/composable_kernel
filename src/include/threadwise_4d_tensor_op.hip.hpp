#pragma once
#include "ConstantTensorDescriptor.hip.hpp"

template <class Float, class Desc, class F>
__device__ void threadwise_4d_tensor_pointwise_operation_unary(Desc, Float* __restrict__ p, F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

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
            for(index_t did2 = 0; did2 < desc.GetLength(I2); ++did2)
            {
                for(index_t did3 = 0; did3 < desc.GetLength(I3); ++did3)
                {
                    const index_t dindex = desc.Get1dIndex(did0, did1, did2, did3);

                    f(p[dindex]);
                }
            }
        }
    }
}

// TODO: in order to optimize mem access for different mem type,
// need to write specialized version
template <class SrcData,
          class DstData,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class MapDst2Src,
          class F>
__device__ void threadwise_4d_tensor_pointwise_operation_binary_reorder_given_dst2src(
    SrcDesc,
    const SrcData* __restrict__ p_src,
    DstDesc,
    DstData* __restrict__ p_dst,
    SrcOpLengths,
    MapDst2Src,
    F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr index_t IR0 = MapDst2Src{}.Get(I0);
    constexpr index_t IR1 = MapDst2Src{}.Get(I1);
    constexpr index_t IR2 = MapDst2Src{}.Get(I2);
    constexpr index_t IR3 = MapDst2Src{}.Get(I3);

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    for(index_t did0 = 0; did0 < ref_desc.GetLength(I0); ++did0)
    {
        for(index_t did1 = 0; did1 < ref_desc.GetLength(I1); ++did1)
        {
            for(index_t did2 = 0; did2 < ref_desc.GetLength(I2); ++did2)
            {
                for(index_t did3 = 0; did3 < ref_desc.GetLength(I3); ++did3)
                {
                    const index_t aindex = src_desc.Get1dIndex(did0, did1, did2, did3);

                    const index_t did[4] = {did0, did1, did2, did3};

                    const index_t bindex =
                        dst_desc.Get1dIndex(did[IR0], did[IR1], did[IR2], did[IR3]);

                    f(p_src[aindex], p_dst[bindex]);

#if 0
                    if(get_block_1d_id() == 0)
                    {
                        printf("tid %5u, "
                               "src did %u %u %u %u, "
                               "dst did %u %u %u %u, "
                               "aindex %5u, "
                               "bindex %5u\n",
                               get_thread_local_1d_id(),
                               did0,
                               did1,
                               did2,
                               did3,
                               did[IR0],
                               did[IR1],
                               did[IR2],
                               did[IR3],
                               aindex,
                               bindex);
                    }
#endif
                }
            }
        }
    }
}

template <class Data, class Desc>
__device__ void threadwise_4d_tensor_set_zero(Desc, Data* __restrict__ p)
{
    auto f_set_zero = [](Data& v) { v = Data(0); };

    threadwise_4d_tensor_pointwise_operation_unary<Data, Desc, decltype(f_set_zero)>(
        Desc{}, p, f_set_zero);
}

template <class SrcData,
          class DstData,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class MapDst2Src>
__device__ void threadwise_4d_tensor_copy_reorder_given_dst2src(SrcDesc,
                                                                const SrcData* __restrict__ p_src,
                                                                DstDesc,
                                                                DstData* __restrict__ p_dst,
                                                                SrcOpLengths,
                                                                MapDst2Src)
{
    auto f_copy = [](const SrcData& src, DstData& dst) { dst = static_cast<DstData>(src); };

    threadwise_4d_tensor_pointwise_operation_binary_reorder_given_dst2src(
        SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, MapDst2Src{}, f_copy);
}

template <class Float, class Desc, class IDim, class NShift>
__device__ void threadwise_4d_tensor_shift_down(Desc, Float* __restrict__ p, IDim, NShift)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

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

    constexpr index_t did2_end =
        is_same<decltype(I2), IDim>::value ? desc.GetLength(I2) - nshift : desc.GetLength(I2);

    constexpr index_t did3_end =
        is_same<decltype(I3), IDim>::value ? desc.GetLength(I3) - nshift : desc.GetLength(I3);

    for(index_t did0 = 0; did0 < did0_end; ++did0)
    {
        for(index_t did1 = 0; did1 < did1_end; ++did1)
        {
            for(index_t did2 = 0; did2 < did2_end; ++did2)
            {
                for(index_t did3 = 0; did3 < did3_end; ++did3)
                {
                    const index_t dindex = desc.Get1dIndex(did0, did1, did2, did3);

                    const index_t sindex = dindex + nshift * desc.GetStride(IDim{});

                    p[dindex] = p[sindex];
                }
            }
        }
    }
}
