#pragma once
#include "ConstantTensorDescriptor.hip.hpp"

// need to assume src and dst is aligned
template <class Float, class SrcDesc, class DstDesc, class SrcOpLengths, index_t DataPerRead>
__device__ void threadwise_6d_tensor_copy(SrcDesc,
                                          const Float* __restrict__ p_src,
                                          DstDesc,
                                          Float* __restrict__ p_dst,
                                          SrcOpLengths,
                                          Number<DataPerRead>)
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    static_assert(SrcDesc{}.GetDimension() == 6 && DstDesc{}.GetDimension() == 6 &&
                      SrcOpLengths::nDim == 6,
                  "wrong! should be 6 dimension");

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    static_assert(SrcDesc{}.GetStride(I5) == 1 && DstDesc{}.GetStride(I5) == 1,
                  "wrong! only support stride5 == 1!\n");

    static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                  "wrong! only support DataPerRead == 1, 2 or 4!\n");

    static_assert(SrcDesc{}.GetStride(I4) % DataPerRead == 0 &&
                      DstDesc{}.GetStride(I4) % DataPerRead == 0,
                  "wrong! src and dst stride should be multiple of DataPerRead to keep alignment");

    constexpr index_t L5 = SrcOpLengths{}.Get(I5);

    static_assert(L5 % DataPerRead == 0, "wrong! L5 should be evenly divided by DataPerRead");

    constexpr index_t nloop_d5 = L5 / DataPerRead;

    for(index_t did0 = 0; did0 < ref_desc.GetLength(I0); ++did0)
    {
        for(index_t did1 = 0; did1 < ref_desc.GetLength(I1); ++did1)
        {
            for(index_t did2 = 0; did2 < ref_desc.GetLength(I2); ++did2)
            {
                for(index_t did3 = 0; did3 < ref_desc.GetLength(I3); ++did3)
                {
                    for(index_t did4 = 0; did4 < ref_desc.GetLength(I4); ++did4)
                    {
                        for(index_t iloop_d5 = 0; iloop_d5 < nloop_d5; ++iloop_d5)
                        {
                            const index_t src_index = src_desc.Get1dIndex(
                                did0, did1, did2, did3, did4, iloop_d5 * DataPerRead);

                            const index_t dst_index = dst_desc.Get1dIndex(
                                did0, did1, did2, did3, did4, iloop_d5 * DataPerRead);

                            *(reinterpret_cast<vector_t*>(p_dst + dst_index)) =
                                *(reinterpret_cast<const vector_t*>(p_src + src_index));
                        }
                    }
                }
            }
        }
    }
}

// need to assume src and dst is aligned
template <class Float, class SrcDesc, class DstDesc, class SrcOpLengths, index_t DataPerRead>
__device__ void threadwise_8d_tensor_copy(SrcDesc,
                                          const Float* __restrict__ p_src,
                                          DstDesc,
                                          Float* __restrict__ p_dst,
                                          SrcOpLengths,
                                          Number<DataPerRead>)
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    static_assert(SrcDesc{}.GetDimension() == 8 && DstDesc{}.GetDimension() == 8 &&
                      SrcOpLengths::nDim == 8,
                  "wrong! should be 8 dimension");

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    static_assert(SrcDesc{}.GetStride(I7) == 1 && DstDesc{}.GetStride(I7) == 1,
                  "wrong! only support stride7 == 1!\n");

    static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                  "wrong! only support DataPerRead == 1, 2 or 4!\n");

    static_assert(SrcDesc{}.GetStride(I6) % DataPerRead == 0 &&
                      DstDesc{}.GetStride(I6) % DataPerRead == 0,
                  "wrong! src and dst stride should be multiple of DataPerRead to keep alignment");

    constexpr index_t L7 = SrcOpLengths{}.Get(I7);

    static_assert(L7 % DataPerRead == 0, "wrong! L7 should be evenly divided by DataPerRead");

    constexpr index_t nloop_d7 = L7 / DataPerRead;

    for(index_t did0 = 0; did0 < ref_desc.GetLength(I0); ++did0)
    {
        for(index_t did1 = 0; did1 < ref_desc.GetLength(I1); ++did1)
        {
            for(index_t did2 = 0; did2 < ref_desc.GetLength(I2); ++did2)
            {
                for(index_t did3 = 0; did3 < ref_desc.GetLength(I3); ++did3)
                {
                    for(index_t did4 = 0; did4 < ref_desc.GetLength(I4); ++did4)
                    {
                        for(index_t did5 = 0; did5 < ref_desc.GetLength(I5); ++did5)
                        {
                            for(index_t did6 = 0; did6 < ref_desc.GetLength(I6); ++did6)
                            {
                                for(index_t iloop_d7 = 0; iloop_d7 < nloop_d7; ++iloop_d7)
                                {
                                    const index_t src_index =
                                        src_desc.Get1dIndex(did0,
                                                            did1,
                                                            did2,
                                                            did3,
                                                            did4,
                                                            did5,
                                                            did6,
                                                            iloop_d7 * DataPerRead);

                                    const index_t dst_index =
                                        dst_desc.Get1dIndex(did0,
                                                            did1,
                                                            did2,
                                                            did3,
                                                            did4,
                                                            did5,
                                                            did6,
                                                            iloop_d7 * DataPerRead);

                                    *(reinterpret_cast<vector_t*>(p_dst + dst_index)) =
                                        *(reinterpret_cast<const vector_t*>(p_src + src_index));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// need to assume src and dst is aligned
template <class Float, class SrcDesc, class DstDesc, class SrcOpLengths, index_t DataPerRead>
__device__ void threadwise_10d_tensor_copy(SrcDesc,
                                           const Float* __restrict__ p_src,
                                           DstDesc,
                                           Float* __restrict__ p_dst,
                                           SrcOpLengths,
                                           Number<DataPerRead>)
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    static_assert(SrcDesc{}.GetDimension() == 10 && DstDesc{}.GetDimension() == 10 &&
                      SrcOpLengths::nDim == 10,
                  "wrong! should be 10 dimension");

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};
    constexpr auto I8 = Number<8>{};
    constexpr auto I9 = Number<9>{};

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    static_assert(SrcDesc{}.GetStride(I9) == 1 && DstDesc{}.GetStride(I9) == 1,
                  "wrong! only support stride7 == 1!\n");

    static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                  "wrong! only support DataPerRead == 1, 2 or 4!\n");

    static_assert(SrcDesc{}.GetStride(I8) % DataPerRead == 0 &&
                      DstDesc{}.GetStride(I8) % DataPerRead == 0,
                  "wrong! src and dst stride should be multiple of DataPerRead to keep alignment");

    constexpr index_t L9 = SrcOpLengths{}.Get(I9);

    static_assert(L9 % DataPerRead == 0, "wrong! L9 should be evenly divided by DataPerRead");

    constexpr index_t nloop_d9 = L9 / DataPerRead;

#pragma unroll
    for(index_t did0 = 0; did0 < ref_desc.GetLength(I0); ++did0)
    {
#pragma unroll
        for(index_t did1 = 0; did1 < ref_desc.GetLength(I1); ++did1)
        {
#pragma unroll
            for(index_t did2 = 0; did2 < ref_desc.GetLength(I2); ++did2)
            {
#pragma unroll
                for(index_t did3 = 0; did3 < ref_desc.GetLength(I3); ++did3)
                {
#pragma unroll
                    for(index_t did4 = 0; did4 < ref_desc.GetLength(I4); ++did4)
                    {
#pragma unroll
                        for(index_t did5 = 0; did5 < ref_desc.GetLength(I5); ++did5)
                        {
#pragma unroll
                            for(index_t did6 = 0; did6 < ref_desc.GetLength(I6); ++did6)
                            {
#pragma unroll
                                for(index_t did7 = 0; did7 < ref_desc.GetLength(I7); ++did7)
                                {
#pragma unroll
                                    for(index_t did8 = 0; did8 < ref_desc.GetLength(I8); ++did8)
                                    {
#pragma unroll
                                        for(index_t iloop_d9 = 0; iloop_d9 < nloop_d9; ++iloop_d9)
                                        {
                                            const index_t src_index =
                                                src_desc.Get1dIndex(did0,
                                                                    did1,
                                                                    did2,
                                                                    did3,
                                                                    did4,
                                                                    did5,
                                                                    did6,
                                                                    did7,
                                                                    did8,
                                                                    iloop_d9 * DataPerRead);

                                            const index_t dst_index =
                                                dst_desc.Get1dIndex(did0,
                                                                    did1,
                                                                    did2,
                                                                    did3,
                                                                    did4,
                                                                    did5,
                                                                    did6,
                                                                    did7,
                                                                    did8,
                                                                    iloop_d9 * DataPerRead);

                                            *(reinterpret_cast<vector_t*>(p_dst + dst_index)) =
                                                *(reinterpret_cast<const vector_t*>(p_src +
                                                                                    src_index));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
