#ifndef CK_BLOCKWISE_2D_TENSOR_OP_HPP
#define CK_BLOCKWISE_2D_TENSOR_OP_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"

namespace ck {

template <index_t BlockSize, class Float, class DstDesc, class F>
__device__ void
blockwise_2d_tensor_pointwise_operation_unary(DstDesc, Float* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    constexpr auto dst_desc = DstDesc{};

    constexpr auto desc = make_ConstantTensorDescriptor(dst_desc.GetLengths());

#if 0
    if(get_thread_local_1d_id() == 0)
    {
        print_ConstantTensorDescriptor(dst_desc, "blockwise_4d_tensor_op_unary: dst_desc: ");
        print_ConstantTensorDescriptor(desc, "blockwise_4d_tensor_op_unary: desc: ");
    }
#endif

    constexpr index_t NLoop = desc.GetElementSize() / BlockSize;

    for(index_t iloop = 0; iloop < NLoop; ++iloop)
    {
        index_t is = get_thread_local_1d_id() + iloop * BlockSize;

        const index_t did0 = is / desc.GetStride(I0);

        is -= did0 * desc.GetStride(I0);

        const index_t did1 = is / desc.GetStride(I1);

        const index_t dindex = dst_desc.GetOffsetFromMultiIndex(did0, did1);

        f(p_dst[dindex]);
    }

    constexpr bool has_tail = (desc.GetElementSize() > NLoop * BlockSize);

    if(has_tail)
    {
        index_t is = get_thread_local_1d_id() + NLoop * BlockSize;

        if(is < desc.GetElementSize())
        {
            const index_t did0 = is / desc.GetStride(I0);

            is -= did0 * desc.GetStride(I0);

            const index_t did1 = is / desc.GetStride(I1);

            const index_t dindex = dst_desc.GetOffsetFromMultiIndex(did0, did1);

            f(p_dst[dindex]);
        }
    }
}

// Function: p_dst[reorder[i0], reorder[i1] = p_src[i0,i1]
// TODO: in order to optimize mem access for different mem type,
// need to write specialized version
template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class MapDst2Src,
          class F>
__device__ void blockwise_2d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src(
    SrcDesc,
    const Float* __restrict__ p_src,
    DstDesc,
    Float* __restrict__ p_dst,
    SrcOpLengths,
    MapDst2Src,
    F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    constexpr index_t IR0 = MapDst2Src{}.Get(I0);
    constexpr index_t IR1 = MapDst2Src{}.Get(I1);

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    constexpr index_t NLoop = ref_desc.GetElementSize() / BlockSize;

    for(index_t iloop = 0; iloop < NLoop; ++iloop)
    {
        index_t is = get_thread_local_1d_id() + iloop * BlockSize;

        index_t did[2];

        did[0] = is / ref_desc.GetStride(I0);

        is -= did[0] * ref_desc.GetStride(I0);

        did[1] = is / ref_desc.GetStride(I1);

        const index_t aindex = src_desc.GetOffsetFromMultiIndex(did[0], did[1]);

        const index_t bindex = dst_desc.GetOffsetFromMultiIndex(did[IR0], did[IR1]);

        f(p_src[aindex], p_dst[bindex]);
    }

    constexpr bool has_tail = (ref_desc.GetElementSize() > NLoop * BlockSize);

    if(has_tail)
    {
        index_t is = get_thread_local_1d_id() + NLoop * BlockSize;

        if(is < ref_desc.GetElementSize())
        {
            index_t did[2];

            did[0] = is / ref_desc.GetStride(I0);

            is -= did[0] * ref_desc.GetStride(I0);

            did[1] = is / ref_desc.GetStride(I1);

            const index_t aindex = src_desc.GetOffsetFromMultiIndex(did[0], did[1]);

            const index_t bindex = dst_desc.GetOffsetFromMultiIndex(did[IR0], did[IR1]);

            f(p_src[aindex], p_dst[bindex]);
        }
    }
}

template <index_t BlockSize, class Float, class DstDesc>
__device__ void blockwise_2d_tensor_set_zero(DstDesc, Float* __restrict__ p_dst)
{
    auto f_set_zero = [](Float& v) { v = Float(0); };

    blockwise_2d_tensor_pointwise_operation_unary<BlockSize>(DstDesc{}, p_dst, f_set_zero);
}

template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class MapDst2Src>
__device__ void
blockwise_2d_tensor_copy_reorder_by_get_dst_from_src(SrcDesc,
                                                     const Float* __restrict__ p_src,
                                                     DstDesc,
                                                     Float* __restrict__ p_dst,
                                                     SrcOpLengths,
                                                     MapDst2Src)
{
    auto f_copy = [](const Float& src, Float& dst) { dst = src; };

    blockwise_2d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src<BlockSize>(
        SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, MapDst2Src{}, f_copy);
}

template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class CopyLengths,
          index_t DataPerRead>
struct Blockwise2dTensorCopy1
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    __device__ constexpr Blockwise2dTensorCopy1()
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        static_assert(DataPerRead == 1 ||
                          (SrcDesc{}.GetStride(I1) == 1 && DstDesc{}.GetStride(I1) == 1),
                      "wrong! only support stride1 == 1 if DataPerRead > 1!\n");

        static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                      "wrong! only support DataPerRead == 1, 2 or 4!\n");

        static_assert(SrcDesc{}.GetStride(I0) % DataPerRead == 0 &&
                          DstDesc{}.GetStride(I0) % DataPerRead == 0,
                      "src and dst stride2 should be multiple of DataPerRead to keep alignment");

        // we allow out-of-bound read from src in D1 dimension,
        //   but we need to make sure dst stride0 is big enough,
        //   so that the out-of-bound write won't contaminate next line in dst
        constexpr index_t L1          = CopyLengths{}.Get(I1);
        constexpr index_t read_per_d1 = math::integer_divide_ceil(L1, DataPerRead);

        static_assert(read_per_d1 * DataPerRead <= DstDesc{}.GetStride(I0),
                      "wrong! out-of-bound write will contaminate next line!\n");
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto src_desc = SrcDesc{};
        constexpr auto dst_desc = DstDesc{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);

        constexpr index_t read_per_d1 = math::integer_divide_ceil(L1, DataPerRead);

        constexpr auto ref_desc = make_ConstantTensorDescriptor(Sequence<L0, read_per_d1>{});

        constexpr index_t NLoop = ref_desc.GetElementSize() / BlockSize;

        auto f_copy = [&](index_t is) {
            index_t did[4];

            did[0] = is / ref_desc.GetStride(I0);

            is -= did[0] * ref_desc.GetStride(I0);

            did[1] = is / ref_desc.GetStride(I1);

            const index_t src_index =
                src_desc.GetOffsetFromMultiIndex(did[0], did[1] * DataPerRead);
            const index_t dst_index =
                dst_desc.GetOffsetFromMultiIndex(did[0], did[1] * DataPerRead);

            *(reinterpret_cast<vector_t*>(p_dst + dst_index)) =
                *(reinterpret_cast<const vector_t*>(p_src + src_index));
        };

        for(index_t iloop = 0; iloop < NLoop; ++iloop)
        {
            index_t is = get_thread_local_1d_id() + iloop * BlockSize;

            f_copy(is);
        }

        constexpr bool has_tail = (ref_desc.GetElementSize() > NLoop * BlockSize);

        if(has_tail)
        {
            index_t is = get_thread_local_1d_id() + NLoop * BlockSize;

            if(is < ref_desc.GetElementSize())
            {
                f_copy(is);
            }
        }
    }
};

// need to be aligned to float4 and float2
// stride1 need to be 1 for both source and destination
template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          index_t ThreadPerDim0,
          index_t ThreadPerDim1>
struct Blockwise2dTensorCopy2
{
    index_t mThreadId0;
    index_t mThreadId1;

    __device__ Blockwise2dTensorCopy2()
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        static_assert(SrcDesc{}.GetStride(I1) == 1 && DstDesc{}.GetStride(I1) == 1,
                      "wrong! stride is not 1!\n");

        mThreadId0 = get_thread_local_1d_id() / ThreadPerDim1;
        mThreadId1 = get_thread_local_1d_id() - mThreadId0 * ThreadPerDim1;
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        static_assert(is_same<Float, float>{}, "wrong! only support float!\n");

        using Float4 = float4;
        using Float2 = float2;

        if(get_thread_local_1d_id() >= ThreadPerDim0 * ThreadPerDim1)
            return;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto src_desc = SrcDesc{};
        constexpr auto dst_desc = DstDesc{};

        // check alignment
        constexpr bool align_v4 =
            src_desc.GetStride(I0) % 4 == 0 && dst_desc.GetStride(I0) % 4 == 0;

        constexpr bool align_v2 =
            src_desc.GetStride(I0) % 2 == 0 && dst_desc.GetStride(I0) % 2 == 0;

        constexpr index_t L0 = SrcOpLengths{}.Get(I0);
        constexpr index_t L1 = SrcOpLengths{}.Get(I1);

        constexpr index_t Dim0Loop = L0 / ThreadPerDim0;
        constexpr bool d0_has_tail = (L0 > ThreadPerDim0 * Dim0Loop);

        constexpr index_t Dim1V4Loop = align_v4 ? L1 / (ThreadPerDim1 * 4) : 0;

        constexpr index_t Dim1V2Loop =
            align_v2 ? (L1 - Dim1V4Loop * (ThreadPerDim1 * 4)) / (ThreadPerDim1 * 2) : 0;

        constexpr index_t Dim1V1Loop =
            (L1 - Dim1V4Loop * (ThreadPerDim1 * 4) - Dim1V2Loop * (ThreadPerDim1 * 2)) /
            ThreadPerDim1;

        constexpr bool d1_has_tail =
            (L1 > ThreadPerDim1 * (4 * Dim1V4Loop + 2 * Dim1V2Loop + Dim1V1Loop));

        for(index_t d0loop = 0; d0loop < Dim0Loop; ++d0loop)
        {
            index_t did0 = d0loop * ThreadPerDim0 + mThreadId0;

            // v4
            for(index_t d1v4loop = 0; d1v4loop < Dim1V4Loop; ++d1v4loop)
            {
                index_t did1 = d1v4loop * 4 * ThreadPerDim1 + 4 * mThreadId1;

                const index_t sindex = src_desc.GetOffsetFromMultiIndex(did0, did1);
                const index_t dindex = dst_desc.GetOffsetFromMultiIndex(did0, did1);

                *(reinterpret_cast<Float4*>(p_dst + dindex)) =
                    *(reinterpret_cast<const Float4*>(p_src + sindex));
            }

            // v2
            for(index_t d1v2loop = 0; d1v2loop < Dim1V2Loop; ++d1v2loop)
            {
                index_t did1 =
                    Dim1V4Loop * 4 * ThreadPerDim1 + d1v2loop * 2 * ThreadPerDim1 + 2 * mThreadId1;

                const index_t sindex = src_desc.GetOffsetFromMultiIndex(did0, did1);
                const index_t dindex = dst_desc.GetOffsetFromMultiIndex(did0, did1);

                *(reinterpret_cast<Float2*>(p_dst + dindex)) =
                    *(reinterpret_cast<const Float2*>(p_src + sindex));
            }

            // v1
            for(index_t d1v1loop = 0; d1v1loop < Dim1V1Loop; ++d1v1loop)
            {
                index_t did1 = Dim1V4Loop * 4 * ThreadPerDim1 + Dim1V2Loop * 2 * ThreadPerDim1 +
                               d1v1loop * ThreadPerDim1 + mThreadId1;

                const index_t sindex = src_desc.GetOffsetFromMultiIndex(did0, did1);
                const index_t dindex = dst_desc.GetOffsetFromMultiIndex(did0, did1);

                p_dst[dindex] = p_src[sindex];
            }

            // dim-1 tail
            if(d1_has_tail)
            {
                index_t did1 = Dim1V4Loop * 4 * ThreadPerDim1 + Dim1V2Loop * 2 * ThreadPerDim1 +
                               Dim1V1Loop * ThreadPerDim1 + mThreadId1;

                if(did1 < L1)
                {
                    const index_t sindex = src_desc.GetOffsetFromMultiIndex(did0, did1);
                    const index_t dindex = dst_desc.GetOffsetFromMultiIndex(did0, did1);

                    p_dst[dindex] = p_src[sindex];
                }
            }
        }

        // dim-0 tail
        if(d0_has_tail)
        {
            index_t did0 = Dim0Loop * ThreadPerDim0 + mThreadId0;

            if(did0 < L0)
            {

                // v4
                for(index_t d1v4loop = 0; d1v4loop < Dim1V4Loop; ++d1v4loop)
                {
                    index_t did1 = d1v4loop * 4 * ThreadPerDim1 + 4 * mThreadId1;

                    const index_t sindex = src_desc.GetOffsetFromMultiIndex(did0, did1);
                    const index_t dindex = dst_desc.GetOffsetFromMultiIndex(did0, did1);

                    *(reinterpret_cast<Float4*>(p_dst + dindex)) =
                        *(reinterpret_cast<const Float4*>(p_src + sindex));
                }

                // v2
                for(index_t d1v2loop = 0; d1v2loop < Dim1V2Loop; ++d1v2loop)
                {
                    index_t did1 = Dim1V4Loop * 4 * ThreadPerDim1 + d1v2loop * 2 * ThreadPerDim1 +
                                   2 * mThreadId1;

                    const index_t sindex = src_desc.GetOffsetFromMultiIndex(did0, did1);
                    const index_t dindex = dst_desc.GetOffsetFromMultiIndex(did0, did1);

                    *(reinterpret_cast<Float2*>(p_dst + dindex)) =
                        *(reinterpret_cast<const Float2*>(p_src + sindex));
                }

                // v1
                for(index_t d1v1loop = 0; d1v1loop < Dim1V1Loop; ++d1v1loop)
                {
                    index_t did1 = Dim1V4Loop * 4 * ThreadPerDim1 + Dim1V2Loop * 2 * ThreadPerDim1 +
                                   d1v1loop * ThreadPerDim1 + mThreadId1;

                    const index_t sindex = src_desc.GetOffsetFromMultiIndex(did0, did1);
                    const index_t dindex = dst_desc.GetOffsetFromMultiIndex(did0, did1);

                    p_dst[dindex] = p_src[sindex];
                }

                // tail
                if(d1_has_tail)
                {
                    index_t did1 = Dim1V4Loop * 4 * ThreadPerDim1 + Dim1V2Loop * 2 * ThreadPerDim1 +
                                   Dim1V1Loop * ThreadPerDim1 + mThreadId1;

                    if(did1 < L1)
                    {
                        const index_t sindex = src_desc.GetOffsetFromMultiIndex(did0, did1);
                        const index_t dindex = dst_desc.GetOffsetFromMultiIndex(did0, did1);

                        p_dst[dindex] = p_src[sindex];
                    }
                }
            }
        }
    }
};

// starting point need to be aligned to float4 or float2 or float
// stride1 need to be 1 for both source and destination
template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class CopyLengths,
          index_t DataPerRead>
struct Blockwise2dTensorCopy3
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    index_t mSrcMyThreadOffset;
    index_t mDstMyThreadOffset;

    __device__ Blockwise2dTensorCopy3(Array<index_t, 2> src_block_data_multi_id_begin,
                                      Array<index_t, 2> dst_block_data_multi_id_begin)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        static_assert(DataPerRead == 1 ||
                          (SrcDesc{}.GetStride(I1) == 1 && DstDesc{}.GetStride(I1) == 1),
                      "wrong! only support stride1 == 1 if DataPerRead > 1!\n");

        static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                      "wrong! only support DataPerRead == 1, 2 or 4!\n");

        static_assert(SrcDesc{}.GetStride(I0) % DataPerRead == 0 &&
                          DstDesc{}.GetStride(I0) % DataPerRead == 0,
                      "src and dst stride should be multiple of DataPerRead to keep alignment");

        constexpr index_t L1 = CopyLengths{}.Get(I1);

        constexpr index_t thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr index_t thread_per_d0 = BlockSize / thread_per_d1;

        // we allow out-of-bound read from src in D1 dimension,
        //   but we need to make sure dst stride is big enough,
        //   so that the out-of-bound write won't contaminate next line in dst
        static_assert(thread_per_d1 * DataPerRead <= DstDesc{}.GetStride(I0),
                      "wrong! out-of-bound write will contaminate next line!\n");

        static_assert(thread_per_d0 >= 1, "wrong! not enough threads to cover one line\n");

        constexpr index_t num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        const index_t thread_id_d0 = get_thread_local_1d_id() / thread_per_d1;
        const index_t thread_id_d1 = get_thread_local_1d_id() - thread_id_d0 * thread_per_d1;

        mSrcMyThreadOffset = SrcDesc{}.GetOffsetFromMultiIndex(
            src_block_data_multi_id_begin +
            Array<index_t, 2>{thread_id_d0, thread_id_d1 * DataPerRead});

        mDstMyThreadOffset = DstDesc{}.GetOffsetFromMultiIndex(
            dst_block_data_multi_id_begin +
            Array<index_t, 2>{thread_id_d0, thread_id_d1 * DataPerRead});
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);

        constexpr index_t thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr index_t thread_per_d0 = BlockSize / thread_per_d1;

        constexpr index_t num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;

        constexpr index_t src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr index_t dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        auto f_copy = [&](index_t iloop) {
            *(reinterpret_cast<vector_t*>(p_dst + mDstMyThreadOffset + iloop * dst_loop_stride)) =
                *(reinterpret_cast<const vector_t*>(p_src + mSrcMyThreadOffset +
                                                    iloop * src_loop_stride));
        };

        for(index_t iloop = 0; iloop < nloop_d0; ++iloop)
        {
            f_copy(iloop);
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr index_t tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                f_copy(nloop_d0);
            }
        }
    }

    __device__ constexpr index_t GetRegisterBufferSize() const
    {
        static_assert(is_same<Float, float>{}, "wrong! only support float!\n");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);

        constexpr index_t thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr index_t thread_per_d0 = BlockSize / thread_per_d1;

        return DataPerRead * (L0 + thread_per_d0 - 1) / thread_per_d0;
    }

    __device__ void RunLoadRegisterBuffer(const Float* __restrict__ p_src,
                                          Float* __restrict__ p_clipboard) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);

        constexpr index_t thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr index_t thread_per_d0 = BlockSize / thread_per_d1;

        constexpr index_t num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;

        constexpr index_t src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr index_t dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        auto f_copy = [&](index_t iloop) {
            *(reinterpret_cast<vector_t*>(&p_clipboard[iloop * DataPerRead])) =
                *(reinterpret_cast<const vector_t*>(
                    &p_src[mSrcMyThreadOffset + iloop * src_loop_stride]));
        };

        for(index_t iloop = 0; iloop < nloop_d0; ++iloop)
        {
            f_copy(iloop);
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr index_t tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                f_copy(nloop_d0);
            }
        }
    }

    __device__ void RunStoreRegisterBuffer(const Float* __restrict__ p_clipboard,
                                           Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);

        constexpr index_t thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr index_t thread_per_d0 = BlockSize / thread_per_d1;

        constexpr index_t num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;

        constexpr index_t src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr index_t dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        auto f_copy = [&](index_t iloop) {
            *(reinterpret_cast<vector_t*>(&p_dst[mDstMyThreadOffset + iloop * dst_loop_stride])) =
                *(reinterpret_cast<const vector_t*>(&p_clipboard[iloop * DataPerRead]));
        };

        for(index_t iloop = 0; iloop < nloop_d0; ++iloop)
        {
            f_copy(iloop);
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr index_t tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                f_copy(nloop_d0);
            }
        }
    }

#if CK_USE_AMD_INLINE_ASM
    __device__ void RunLoadRegisterBuffer_asm(const Float* __restrict__ p_src,
                                              Float* p_clipboard) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);

        constexpr index_t thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr index_t thread_per_d0 = BlockSize / thread_per_d1;

        constexpr index_t num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;

        constexpr index_t src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr index_t dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        auto f_copy = [&](index_t iloop) {
#if 0
            *(reinterpret_cast<vector_t*>(&p_clipboard[iloop * DataPerRead])) =
                *(reinterpret_cast<const vector_t*>(&p_src[mSrcMyThreadOffset +
                                                    iloop * src_loop_stride]));
#else
            static_assert(is_same<float, Float>{} && DataPerRead == 4,
                          "global_load is only for float4");

            global_load(reinterpret_cast<vector_t&>(p_clipboard[iloop * DataPerRead]),
                        reinterpret_cast<const vector_t*>(
                            &p_src[mSrcMyThreadOffset + iloop * src_loop_stride]));
#endif
        };

        for(index_t iloop = 0; iloop < nloop_d0; ++iloop)
        {
            f_copy(iloop);
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr index_t tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                f_copy(nloop_d0);
            }
        }
    }

    __device__ void RunStoreRegisterBuffer_asm(const Float* __restrict__ p_clipboard,
                                               Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);

        constexpr index_t thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr index_t thread_per_d0 = BlockSize / thread_per_d1;

        constexpr index_t num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;

        constexpr index_t src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr index_t dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        auto f_copy = [&](index_t iloop) {
#if 0
            *(reinterpret_cast<vector_t*>(&p_dst[mDstMyThreadOffset + iloop * dst_loop_stride]) =
                *(reinterpret_cast<const vector_t*>(&p_clipboard[iloop * DataPerRead]);
#else
            static_assert(is_same<float, Float>{} && DataPerRead == 4,
                          "ds_write_b128 is only for float4");

            ds_write_b128(reinterpret_cast<const vector_t&>(p_clipboard[iloop * DataPerRead]),
                          &p_dst[mDstMyThreadOffset + iloop * dst_loop_stride]);
#endif
        };

        for(index_t iloop = 0; iloop < nloop_d0; ++iloop)
        {
            f_copy(iloop);
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr index_t tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                f_copy(nloop_d0);
            }
        }
    }
#endif
};

} // namespace ck

#endif
