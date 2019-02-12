#pragma once
#include "ConstantTensorDescriptor.cuh"

template <unsigned BlockSize, class Float, class DstDesc, class F>
__device__ void
blockwise_2d_tensor_pointwise_operation_unary(DstDesc, Float* __restrict__ p_dst, F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

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

        const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

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

            const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

            f(p_dst[dindex]);
        }
    }
}

// Function: p_dst[reorder[i0], reorder[i1], reorder[i2], reorder[i3]] = p_src[i0,i1,i2,i3]
// TODO: in order to optimize mem access for different mem type,
// need to write specialized version
template <unsigned BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class DstFromSrcReorder,
          class F>
__device__ void blockwise_2d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src(
    SrcDesc,
    const Float* __restrict__ p_src,
    DstDesc,
    Float* __restrict__ p_dst,
    SrcOpLengths,
    DstFromSrcReorder,
    F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    constexpr unsigned IR0 = DstFromSrcReorder{}.Get(I0);
    constexpr unsigned IR1 = DstFromSrcReorder{}.Get(I1);

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    constexpr unsigned NLoop = ref_desc.GetElementSize() / BlockSize;

    for(unsigned iloop = 0; iloop < NLoop; ++iloop)
    {
        unsigned is = threadIdx.x + iloop * BlockSize;

        unsigned did[2];

        did[0] = is / ref_desc.GetStride(I0);

        is -= did[0] * ref_desc.GetStride(I0);

        did[1] = is / ref_desc.GetStride(I1);

        const unsigned aindex = src_desc.Get1dIndex(did[0], did[1]);

        const unsigned bindex = dst_desc.Get1dIndex(did[IR0], did[IR1]);

        f(p_src[aindex], p_dst[bindex]);
    }

    constexpr bool has_tail = (ref_desc.GetElementSize() > NLoop * BlockSize);

    if(has_tail)
    {
        unsigned is = threadIdx.x + NLoop * BlockSize;

        if(is < ref_desc.GetElementSize())
        {
            unsigned did[2];

            did[0] = is / ref_desc.GetStride(I0);

            is -= did[0] * ref_desc.GetStride(I0);

            did[1] = is / ref_desc.GetStride(I1);

            const unsigned aindex = src_desc.Get1dIndex(did[0], did[1]);

            const unsigned bindex = dst_desc.Get1dIndex(did[IR0], did[IR1]);

            f(p_src[aindex], p_dst[bindex]);
        }
    }
}

template <unsigned BlockSize, class Float, class DstDesc>
__device__ void blockwise_2d_tensor_set_zero(DstDesc, Float* __restrict__ p_dst)
{
    auto f_set_zero = [](Float& v) { v = Float(0); };

    blockwise_2d_tensor_pointwise_operation_unary<BlockSize>(DstDesc{}, p_dst, f_set_zero);
}

template <unsigned BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class DstFromSrcReorder>
__device__ void
blockwise_2d_tensor_copy_reorder_by_get_dst_from_src(SrcDesc,
                                                     const Float* __restrict__ p_src,
                                                     DstDesc,
                                                     Float* __restrict__ p_dst,
                                                     SrcOpLengths,
                                                     DstFromSrcReorder)
{
    auto f_copy = [](const Float& src, Float& dst) { dst = src; };

    blockwise_2d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src<BlockSize>(
        SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, DstFromSrcReorder{}, f_copy);
}

template <unsigned BlockSize, class Float, class SrcDesc, class DstDesc, class SrcOpLengths>
struct Blockwise2dTensorCopy1
{
    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr auto dst_from_src_reorder = Sequence<0, 1>{};

        blockwise_2d_tensor_copy_reorder_by_get_dst_from_src<BlockSize>(
            SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, dst_from_src_reorder);
    }
};

// need to be aligned to float4 and float2
// stride1 need to be 1 for both source and destination
template <unsigned BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          unsigned ThreadPerDim0,
          unsigned ThreadPerDim1>
struct Blockwise2dTensorCopy2
{
    unsigned mThreadId0;
    unsigned mThreadId1;

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
        static_assert(is_same<Float, float>::value, "wrong! only support float!\n");

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

        constexpr unsigned L0 = SrcOpLengths{}.Get(I0);
        constexpr unsigned L1 = SrcOpLengths{}.Get(I1);

        constexpr unsigned Dim0Loop = L0 / ThreadPerDim0;
        constexpr bool d0_has_tail  = (L0 > ThreadPerDim0 * Dim0Loop);

        constexpr unsigned Dim1V4Loop = align_v4 ? L1 / (ThreadPerDim1 * 4) : 0;

        constexpr unsigned Dim1V2Loop =
            align_v2 ? (L1 - Dim1V4Loop * (ThreadPerDim1 * 4)) / (ThreadPerDim1 * 2) : 0;

        constexpr unsigned Dim1V1Loop =
            (L1 - Dim1V4Loop * (ThreadPerDim1 * 4) - Dim1V2Loop * (ThreadPerDim1 * 2)) /
            ThreadPerDim1;

        constexpr bool d1_has_tail =
            (L1 > ThreadPerDim1 * (4 * Dim1V4Loop + 2 * Dim1V2Loop + Dim1V1Loop));

        for(unsigned d0loop = 0; d0loop < Dim0Loop; ++d0loop)
        {
            unsigned did0 = d0loop * ThreadPerDim0 + mThreadId0;

            // v4
            for(unsigned d1v4loop = 0; d1v4loop < Dim1V4Loop; ++d1v4loop)
            {
                unsigned did1 = d1v4loop * 4 * ThreadPerDim1 + 4 * mThreadId1;

                const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                *(reinterpret_cast<Float4*>(p_dst + dindex)) =
                    *(reinterpret_cast<const Float4*>(p_src + sindex));
            }

            // v2
            for(unsigned d1v2loop = 0; d1v2loop < Dim1V2Loop; ++d1v2loop)
            {
                unsigned did1 =
                    Dim1V4Loop * 4 * ThreadPerDim1 + d1v2loop * 2 * ThreadPerDim1 + 2 * mThreadId1;

                const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                *(reinterpret_cast<Float2*>(p_dst + dindex)) =
                    *(reinterpret_cast<const Float2*>(p_src + sindex));
            }

            // v1
            for(unsigned d1v1loop = 0; d1v1loop < Dim1V1Loop; ++d1v1loop)
            {
                unsigned did1 = Dim1V4Loop * 4 * ThreadPerDim1 + Dim1V2Loop * 2 * ThreadPerDim1 +
                                d1v1loop * ThreadPerDim1 + mThreadId1;

                const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                p_dst[dindex] = p_src[sindex];
            }

            // dim-1 tail
            if(d1_has_tail)
            {
                unsigned did1 = Dim1V4Loop * 4 * ThreadPerDim1 + Dim1V2Loop * 2 * ThreadPerDim1 +
                                Dim1V1Loop * ThreadPerDim1 + mThreadId1;

                if(did1 < L1)
                {
                    const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                    p_dst[dindex] = p_src[sindex];
                }
            }
        }

        // dim-0 tail
        if(d0_has_tail)
        {
            unsigned did0 = Dim0Loop * ThreadPerDim0 + mThreadId0;

            if(did0 < L0)
            {

                // v4
                for(unsigned d1v4loop = 0; d1v4loop < Dim1V4Loop; ++d1v4loop)
                {
                    unsigned did1 = d1v4loop * 4 * ThreadPerDim1 + 4 * mThreadId1;

                    const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                    *(reinterpret_cast<Float4*>(p_dst + dindex)) =
                        *(reinterpret_cast<const Float4*>(p_src + sindex));
                }

                // v2
                for(unsigned d1v2loop = 0; d1v2loop < Dim1V2Loop; ++d1v2loop)
                {
                    unsigned did1 = Dim1V4Loop * 4 * ThreadPerDim1 + d1v2loop * 2 * ThreadPerDim1 +
                                    2 * mThreadId1;

                    const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                    *(reinterpret_cast<Float2*>(p_dst + dindex)) =
                        *(reinterpret_cast<const Float2*>(p_src + sindex));
                }

                // v1
                for(unsigned d1v1loop = 0; d1v1loop < Dim1V1Loop; ++d1v1loop)
                {
                    unsigned did1 = Dim1V4Loop * 4 * ThreadPerDim1 +
                                    Dim1V2Loop * 2 * ThreadPerDim1 + d1v1loop * ThreadPerDim1 +
                                    mThreadId1;

                    const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                    p_dst[dindex] = p_src[sindex];
                }

                // tail
                if(d1_has_tail)
                {
                    unsigned did1 = Dim1V4Loop * 4 * ThreadPerDim1 +
                                    Dim1V2Loop * 2 * ThreadPerDim1 + Dim1V1Loop * ThreadPerDim1 +
                                    mThreadId1;

                    if(did1 < L1)
                    {
                        const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                        const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                        p_dst[dindex] = p_src[sindex];
                    }
                }
            }
        }
    }
};

// starting point need to be aligned to float4 or float2 or float
// stride1 need to be 1 for both source and destination
template <unsigned BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class CopyLengths,
          unsigned DataPerRead>
struct Blockwise2dTensorCopy3
{
    unsigned mSrcMyThreadOffset;
    unsigned mDstMyThreadOffset;

    __device__ Blockwise2dTensorCopy3()
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        static_assert(SrcDesc{}.GetStride(I1) == 1 && DstDesc{}.GetStride(I1) == 1,
                      "wrong! only support stride1 == 1!\n");

        static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                      "wrong! only support DataPerRead == 1, 2 or 4!\n");

        static_assert(SrcDesc{}.GetStride(I0) % DataPerRead == 0 &&
                          DstDesc{}.GetStride(I0) % DataPerRead == 0,
                      "src and dst stride should be multiple of DataPerRead to keep alignment");

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);

        constexpr unsigned thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr unsigned thread_per_d0 = BlockSize / thread_per_d1;

        // we allow out-of-bound read from src in D1 dimension,
        //   but we need to make sure dst stride is big enough,
        //   so that the out-of-bound write won't contaminate next line in dst
        static_assert(thread_per_d1 * DataPerRead <= DstDesc{}.GetStride(I0),
                      "wrong! out-of-bound write will contaminate next line!\n");

        static_assert(thread_per_d0 >= 1, "wrong! not enough threads to cover one line\n");

        constexpr unsigned num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        const unsigned thread_id_d0 = get_thread_local_1d_id() / thread_per_d1;
        const unsigned thread_id_d1 = get_thread_local_1d_id() - thread_id_d0 * thread_per_d1;

        mSrcMyThreadOffset = SrcDesc{}.Get1dIndex(thread_id_d0, thread_id_d1 * DataPerRead);
        mDstMyThreadOffset = DstDesc{}.Get1dIndex(thread_id_d0, thread_id_d1 * DataPerRead);
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        static_assert(is_same<Float, float>::value, "wrong! only support float!\n");

        using Float2 = float2;
        using Float4 = float4;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);

        constexpr unsigned thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr unsigned thread_per_d0 = BlockSize / thread_per_d1;

        constexpr unsigned num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr unsigned nloop_d0 = L0 / thread_per_d0;

        constexpr unsigned src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr unsigned dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        for(unsigned iloop = 0; iloop < nloop_d0; ++iloop)
        {
            if(DataPerRead == 1)
            {
                p_dst[mDstMyThreadOffset + iloop * dst_loop_stride] =
                    p_src[mSrcMyThreadOffset + iloop * src_loop_stride];
            }
            else if(DataPerRead == 2)
            {
                *(reinterpret_cast<Float2*>(p_dst + mDstMyThreadOffset + iloop * dst_loop_stride)) =
                    *(reinterpret_cast<const Float2*>(p_src + mSrcMyThreadOffset +
                                                      iloop * src_loop_stride));
            }
            else if(DataPerRead == 4)
            {
                *(reinterpret_cast<Float4*>(p_dst + mDstMyThreadOffset + iloop * dst_loop_stride)) =
                    *(reinterpret_cast<const Float4*>(p_src + mSrcMyThreadOffset +
                                                      iloop * src_loop_stride));
            }
            else
            {
                assert(false);
            }
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr unsigned tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                if(DataPerRead == 1)
                {
                    p_dst[mDstMyThreadOffset + nloop_d0 * dst_loop_stride] =
                        p_src[mSrcMyThreadOffset + nloop_d0 * src_loop_stride];
                }
                else if(DataPerRead == 2)
                {
                    *(reinterpret_cast<Float2*>(p_dst + mDstMyThreadOffset +
                                                nloop_d0 * dst_loop_stride)) =
                        *(reinterpret_cast<const Float2*>(p_src + mSrcMyThreadOffset +
                                                          nloop_d0 * src_loop_stride));
                }
                else if(DataPerRead == 4)
                {
                    *(reinterpret_cast<Float4*>(p_dst + mDstMyThreadOffset +
                                                nloop_d0 * dst_loop_stride)) =
                        *(reinterpret_cast<const Float4*>(p_src + mSrcMyThreadOffset +
                                                          nloop_d0 * src_loop_stride));
                }
                else
                {
                    assert(false);
                }
            }
        }
    }

#if 1
    __device__ constexpr unsigned GetRegisterClipboardSize() const
    {
        static_assert(is_same<Float, float>::value, "wrong! only support float!\n");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);

        constexpr unsigned thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr unsigned thread_per_d0 = BlockSize / thread_per_d1;

        return DataPerRead * (L0 + thread_per_d0 - 1) / thread_per_d0;
    }

    __device__ void RunLoadRegisterClipboard(const Float* __restrict__ p_src,
                                             Float* p_clipboard) const
    {
        static_assert(is_same<Float, float>::value, "wrong! only support float!\n");

        using Float2 = float2;
        using Float4 = float4;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);

        constexpr unsigned thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr unsigned thread_per_d0 = BlockSize / thread_per_d1;

        constexpr unsigned num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr unsigned nloop_d0 = L0 / thread_per_d0;

        constexpr unsigned src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr unsigned dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        for(unsigned iloop = 0; iloop < nloop_d0; ++iloop)
        {
            if(DataPerRead == 1)
            {
                p_clipboard[iloop] = p_src[mSrcMyThreadOffset + iloop * src_loop_stride];
            }
            else if(DataPerRead == 2)
            {
                *(reinterpret_cast<Float2*>(p_clipboard + iloop * 2)) =
                    *(reinterpret_cast<const Float2*>(p_src + mSrcMyThreadOffset +
                                                      iloop * src_loop_stride));
            }
            else if(DataPerRead == 4)
            {
                *(reinterpret_cast<Float4*>(p_clipboard + iloop * 4)) =
                    *(reinterpret_cast<const Float4*>(p_src + mSrcMyThreadOffset +
                                                      iloop * src_loop_stride));
            }
            else
            {
                assert(false);
            }
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr unsigned tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                if(DataPerRead == 1)
                {
                    p_clipboard[nloop_d0] = p_src[mSrcMyThreadOffset + nloop_d0 * src_loop_stride];
                }
                else if(DataPerRead == 2)
                {
                    *(reinterpret_cast<Float2*>(p_clipboard + nloop_d0 * 2)) =
                        *(reinterpret_cast<const Float2*>(p_src + mSrcMyThreadOffset +
                                                          nloop_d0 * src_loop_stride));
                }
                else if(DataPerRead == 4)
                {
                    *(reinterpret_cast<Float4*>(p_clipboard + nloop_d0 * 4)) =
                        *(reinterpret_cast<const Float4*>(p_src + mSrcMyThreadOffset +
                                                          nloop_d0 * src_loop_stride));
                }
                else
                {
                    assert(false);
                }
            }
        }
    }

    __device__ void RunStoreRegisterClipboard(const Float* __restrict__ p_clipboard,
                                              Float* __restrict__ p_dst) const
    {
        static_assert(is_same<Float, float>::value, "wrong! only support float!\n");

        using Float2 = float2;
        using Float4 = float4;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);

        constexpr unsigned thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr unsigned thread_per_d0 = BlockSize / thread_per_d1;

        constexpr unsigned num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr unsigned nloop_d0 = L0 / thread_per_d0;

        constexpr unsigned src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr unsigned dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        for(unsigned iloop = 0; iloop < nloop_d0; ++iloop)
        {
            if(DataPerRead == 1)
            {
                p_dst[mDstMyThreadOffset + iloop * dst_loop_stride] = p_clipboard[iloop];
            }
            else if(DataPerRead == 2)
            {
                *(reinterpret_cast<Float2*>(p_dst + mDstMyThreadOffset + iloop * dst_loop_stride)) =
                    *(reinterpret_cast<const Float2*>(p_clipboard + iloop * 2));
            }
            else if(DataPerRead == 4)
            {
                *(reinterpret_cast<Float4*>(p_dst + mDstMyThreadOffset + iloop * dst_loop_stride)) =
                    *(reinterpret_cast<const Float4*>(p_clipboard + iloop * 4));
            }
            else
            {
                assert(false);
            }
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr unsigned tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                if(DataPerRead == 1)
                {
                    p_dst[mDstMyThreadOffset + nloop_d0 * dst_loop_stride] = p_clipboard[nloop_d0];
                }
                else if(DataPerRead == 2)
                {
                    *(reinterpret_cast<Float2*>(p_dst + mDstMyThreadOffset +
                                                nloop_d0 * dst_loop_stride)) =
                        *(reinterpret_cast<const Float2*>(p_clipboard + nloop_d0 * 2));
                }
                else if(DataPerRead == 4)
                {
                    *(reinterpret_cast<Float4*>(p_dst + mDstMyThreadOffset +
                                                nloop_d0 * dst_loop_stride)) =
                        *(reinterpret_cast<const Float4*>(p_clipboard + nloop_d0 * 4));
                }
                else
                {
                    assert(false);
                }
            }
        }
    }
#endif
};
