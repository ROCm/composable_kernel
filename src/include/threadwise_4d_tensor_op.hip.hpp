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
          class DstFromSrcReorder,
          class F>
__device__ void threadwise_4d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src(
    SrcDesc,
    const SrcData* __restrict__ p_src,
    DstDesc,
    DstData* __restrict__ p_dst,
    SrcOpLengths,
    DstFromSrcReorder,
    F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr unsigned IR0 = DstFromSrcReorder{}.Get(I0);
    constexpr unsigned IR1 = DstFromSrcReorder{}.Get(I1);
    constexpr unsigned IR2 = DstFromSrcReorder{}.Get(I2);
    constexpr unsigned IR3 = DstFromSrcReorder{}.Get(I3);

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    for(unsigned did0 = 0; did0 < ref_desc.GetLength(I0); ++did0)
    {
        for(unsigned did1 = 0; did1 < ref_desc.GetLength(I1); ++did1)
        {
            for(unsigned did2 = 0; did2 < ref_desc.GetLength(I2); ++did2)
            {
                for(unsigned did3 = 0; did3 < ref_desc.GetLength(I3); ++did3)
                {
                    const unsigned aindex = src_desc.Get1dIndex(did0, did1, did2, did3);

                    const unsigned did[4] = {did0, did1, did2, did3};

                    const unsigned bindex =
                        dst_desc.Get1dIndex(did[IR0], did[IR1], did[IR2], did[IR3]);

                    f(p_src[aindex], p_dst[bindex]);
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
          class DstFromSrcReorder>
__device__ void
threadwise_4d_tensor_copy_reorder_by_get_dst_from_src(SrcDesc,
                                                      const SrcData* __restrict__ p_src,
                                                      DstDesc,
                                                      DstData* __restrict__ p_dst,
                                                      SrcOpLengths,
                                                      DstFromSrcReorder)
{
    auto f_copy = [](const SrcData& src, DstData& dst) { dst = static_cast<DstData>(src); };

    threadwise_4d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src(
        SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, DstFromSrcReorder{}, f_copy);
}

template <class SrcData, class DstData, class SrcDesc, class DstDesc, class SrcOpLengths>
__device__ void threadwise_4d_tensor_copy(
    SrcDesc, const SrcData* __restrict__ p_src, DstDesc, DstData* __restrict__ p_dst, SrcOpLengths)
{
    auto dst_from_src_reorder = Sequence<0, 1, 2, 3>{};

    threadwise_4d_tensor_copy_reorder_by_get_dst_from_src(
        SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, dst_from_src_reorder);
}

// need to assume src and dst is aligned
template <class Float, class SrcDesc, class DstDesc, class SrcOpLengths, unsigned DataPerRead>
__device__ void threadwise_4d_tensor_copy_v2(SrcDesc,
                                             const Float* __restrict__ p_src,
                                             DstDesc,
                                             Float* __restrict__ p_dst,
                                             SrcOpLengths,
                                             Number<DataPerRead>)
{
    using Float2 = float2;
    using Float4 = float4;

    static_assert(SrcDesc{}.GetDimension() == 4 && DstDesc{}.GetDimension() == 4 &&
                      SrcOpLengths::nDim == 4,
                  "wrong! should be 4 dimension");

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    static_assert(SrcDesc{}.GetStride(I3) == 1 && DstDesc{}.GetStride(I3) == 1,
                  "wrong! only support stride3 == 1!\n");

    static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                  "wrong! only support DataPerRead == 1, 2 or 4!\n");

    static_assert(SrcDesc{}.GetStride(I2) % DataPerRead == 0 &&
                      DstDesc{}.GetStride(I2) % DataPerRead == 0,
                  "wrong! src and dst stride should be multiple of DataPerRead to keep alignment");

    constexpr unsigned L3 = SrcOpLengths{}.Get(I3);

    static_assert(L3 % DataPerRead == 0, "wrong! L3 should be evenly divided by DataPerRead");

    constexpr unsigned nloop_d3 = L3 / DataPerRead;

    for(unsigned did0 = 0; did0 < ref_desc.GetLength(I0); ++did0)
    {
        for(unsigned did1 = 0; did1 < ref_desc.GetLength(I1); ++did1)
        {
            for(unsigned did2 = 0; did2 < ref_desc.GetLength(I2); ++did2)
            {
                for(unsigned iloop_d3 = 0; iloop_d3 < nloop_d3; ++iloop_d3)
                {
                    const unsigned src_index =
                        src_desc.Get1dIndex(did0, did1, did2, iloop_d3 * DataPerRead);

                    const unsigned dst_index =
                        dst_desc.Get1dIndex(did0, did1, did2, iloop_d3 * DataPerRead);

                    if(DataPerRead == 1)
                    {
                        p_dst[dst_index] = p_src[src_index];
                    }
                    else if(DataPerRead == 2)
                    {
                        *(reinterpret_cast<Float2*>(p_dst + dst_index)) =
                            *(reinterpret_cast<const Float2*>(p_src + src_index));
                    }
                    else if(DataPerRead == 4)
                    {
                        *(reinterpret_cast<Float4*>(p_dst + dst_index)) =
                            *(reinterpret_cast<const Float4*>(p_src + src_index));
                    }
                    else
                    {
                        assert(false);
                    }
                }
            }
        }
    }
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
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(desc, "threadwise_4d_tensor_shift_down: ");
    }
#endif

    constexpr unsigned nshift = NShift::mValue;

    constexpr unsigned did0_end =
        is_same<decltype(I0), IDim>::value ? desc.GetLength(I0) - nshift : desc.GetLength(I0);

    constexpr unsigned did1_end =
        is_same<decltype(I1), IDim>::value ? desc.GetLength(I1) - nshift : desc.GetLength(I1);

    constexpr unsigned did2_end =
        is_same<decltype(I2), IDim>::value ? desc.GetLength(I2) - nshift : desc.GetLength(I2);

    constexpr unsigned did3_end =
        is_same<decltype(I3), IDim>::value ? desc.GetLength(I3) - nshift : desc.GetLength(I3);

    for(unsigned did0 = 0; did0 < did0_end; ++did0)
    {
        for(unsigned did1 = 0; did1 < did1_end; ++did1)
        {
            for(unsigned did2 = 0; did2 < did2_end; ++did2)
            {
                for(unsigned did3 = 0; did3 < did3_end; ++did3)
                {
                    const unsigned dindex = desc.Get1dIndex(did0, did1, did2, did3);

                    const unsigned sindex = dindex + nshift * desc.GetStride(IDim{});

                    p[dindex] = p[sindex];
                }
            }
        }
    }
}
