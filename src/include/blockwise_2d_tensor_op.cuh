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
    Float* const __restrict__ p_src,
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
                                                     Float* const __restrict__ p_src,
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
struct blockwise_2d_tensor_copy_1
{
    __device__ void run(Float* const __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr auto dst_from_src_reorder = Sequence<0, 1>{};

        blockwise_2d_tensor_copy_reorder_by_get_dst_from_src<BlockSize>(
            SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, dst_from_src_reorder);
    }
};

template <unsigned BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          unsigned ThreadPerDim0,
          unsigned ThreadPerDim1>
struct blockwise_2d_tensor_copy_2
{
    unsigned mThreadId0;
    unsigned mThreadId1;

    __device__ blockwise_2d_tensor_copy_2()
    {
        static_assert(is_same<Float, float>::value, "wrong! type is not float!\n");

        mThreadId0 = get_thread_local_1d_id() / ThreadPerDim1;
        mThreadId1 = get_thread_local_1d_id() - mThreadId0 * ThreadPerDim1;
    }

    __device__ void run(Float* const __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        if(get_thread_local_1d_id() >= ThreadPerDim0 * ThreadPerDim1)
            return;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto src_desc = SrcDesc{};
        constexpr auto dst_desc = DstDesc{};

        constexpr unsigned L0 = SrcOpLengths{}.Get(I0);
        constexpr unsigned L1 = SrcOpLengths{}.Get(I1);

        constexpr unsigned Dim0Loop = L0 / ThreadPerDim0;
        constexpr bool d0_has_tail  = (L0 > ThreadPerDim0 * Dim0Loop);

        constexpr unsigned Dim1V4Loop = L1 / (ThreadPerDim1 * 4);
        constexpr unsigned Dim1V2Loop =
            (L1 - Dim1V4Loop * (ThreadPerDim1 * 4)) / (ThreadPerDim1 * 2);
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
#if 1
                const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                *(reinterpret_cast<float4*>(p_dst + dindex)) =
                    *(reinterpret_cast<float4*>(p_src + sindex));

#else
                for(unsigned i = 0; i < 4; ++i)
                {
                    const unsigned sindex = src_desc.Get1dIndex(did0, did1 + i);
                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1 + i);

                    p_dst[dindex] = p_src[sindex];
                }
#endif
            }

            // v2
            for(unsigned d1v2loop = 0; d1v2loop < Dim1V2Loop; ++d1v2loop)
            {
                unsigned did1 =
                    Dim1V4Loop * 4 * ThreadPerDim1 + d1v2loop * 2 * ThreadPerDim1 + 2 * mThreadId1;

#if 1
                const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                *(reinterpret_cast<float2*>(p_dst + dindex)) =
                    *(reinterpret_cast<float2*>(p_src + sindex));

#else
                for(unsigned i = 0; i < 2; ++i)
                {
                    const unsigned sindex = src_desc.Get1dIndex(did0, did1 + i);
                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1 + i);

                    p_dst[dindex] = p_src[sindex];
                }
#endif
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

#if 1
                    const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                    *(reinterpret_cast<float4*>(p_dst + dindex)) =
                        *(reinterpret_cast<float4*>(p_src + sindex));

#else
                    for(unsigned i = 0; i < 4; ++i)
                    {
                        const unsigned sindex = src_desc.Get1dIndex(did0, did1 + i);
                        const unsigned dindex = dst_desc.Get1dIndex(did0, did1 + i);

                        p_dst[dindex] = p_src[sindex];
                    }
#endif
                }

                // v2
                for(unsigned d1v2loop = 0; d1v2loop < Dim1V2Loop; ++d1v2loop)
                {
                    unsigned did1 = Dim1V4Loop * 4 * ThreadPerDim1 + d1v2loop * 2 * ThreadPerDim1 +
                                    2 * mThreadId1;

#if 1
                    const unsigned sindex = src_desc.Get1dIndex(did0, did1);
                    const unsigned dindex = dst_desc.Get1dIndex(did0, did1);

                    *(reinterpret_cast<float2*>(p_dst + dindex)) =
                        *(reinterpret_cast<float2*>(p_src + sindex));

#else
                    for(unsigned i = 0; i < 2; ++i)
                    {
                        const unsigned sindex = src_desc.Get1dIndex(did0, did1 + i);
                        const unsigned dindex = dst_desc.Get1dIndex(did0, did1 + i);

                        p_dst[dindex] = p_src[sindex];
                    }
#endif
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

template <unsigned BlockSize, class Float, class SrcDesc, class DstDesc, class SrcOpLengths>
struct blockwise_2d_tensor_copy_dummy_1
{
    unsigned mBegin;

    __device__ blockwise_2d_tensor_copy_dummy_1()
    {
        constexpr unsigned n_total =
            make_ConstantTensorDescriptor(SrcOpLengths{}).GetElementSpace();

        constexpr unsigned n_per_thread = n_total / BlockSize;

        mBegin = n_per_thread * get_thread_local_1d_id();
    }

    __device__ void run(Float* const __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr unsigned n_total =
            make_ConstantTensorDescriptor(SrcOpLengths{}).GetElementSpace();

        constexpr unsigned n_per_thread = n_total / BlockSize;

        for(unsigned i = 0; i < n_per_thread; ++i)
        {
            p_dst[mBegin + i] = p_src[mBegin + i];
        }
    }
};

template <unsigned BlockSize, class Float, class SrcDesc, class DstDesc, class SrcOpLengths>
struct blockwise_2d_tensor_copy_dummy_2
{
    __device__ void run(Float* const __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr unsigned n_total =
            make_ConstantTensorDescriptor(SrcOpLengths{}).GetElementSpace();

        constexpr unsigned n_per_thread = n_total / BlockSize;

        for(unsigned i = 0; i < n_per_thread; ++i)
        {
            unsigned index = get_thread_local_1d_id() + BlockSize * i;
            p_dst[index]   = p_src[index];
        }
    }
};
