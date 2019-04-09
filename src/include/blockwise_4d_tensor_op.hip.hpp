#pragma once
#include "ConstantTensorDescriptor.hip.hpp"

template <index_t BlockSize, class Float, class DstDesc, class F>
__device__ void
blockwise_4d_tensor_pointwise_operation_unary(DstDesc, Float* __restrict__ p_dst, F f)
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

    constexpr index_t NLoop = desc.GetElementSize() / BlockSize;

    for(index_t iloop = 0; iloop < NLoop; ++iloop)
    {
        index_t is = threadIdx.x + iloop * BlockSize;

        const index_t did0 = is / desc.GetStride(I0);

        is -= did0 * desc.GetStride(I0);

        const index_t did1 = is / desc.GetStride(I1);

        is -= did1 * desc.GetStride(I1);

        const index_t did2 = is / desc.GetStride(I2);

        is -= did2 * desc.GetStride(I2);

        const index_t did3 = is / desc.GetStride(I3);

        const index_t dindex = dst_desc.Get1dIndex(did0, did1, did2, did3);

        f(p_dst[dindex]);
    }

    constexpr bool has_tail = (desc.GetElementSize() > NLoop * BlockSize);

    if(has_tail)
    {
        index_t is = threadIdx.x + NLoop * BlockSize;

        if(is < desc.GetElementSize())
        {
            const index_t did0 = is / desc.GetStride(I0);

            is -= did0 * desc.GetStride(I0);

            const index_t did1 = is / desc.GetStride(I1);

            is -= did1 * desc.GetStride(I1);

            const index_t did2 = is / desc.GetStride(I2);

            is -= did2 * desc.GetStride(I2);

            const index_t did3 = is / desc.GetStride(I3);

            const index_t dindex = dst_desc.Get1dIndex(did0, did1, did2, did3);

            f(p_dst[dindex]);
        }
    }
}

// Function: p_dst[reorder[i0], reorder[i1], reorder[i2], reorder[i3]] = p_src[i0,i1,i2,i3]
// TODO: in order to optimize mem access for different mem type,
// need to write specialized version
template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class DstFromSrcReorder,
          class F>
__device__ void blockwise_4d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src(
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
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr index_t IR0 = DstFromSrcReorder{}.Get(I0);
    constexpr index_t IR1 = DstFromSrcReorder{}.Get(I1);
    constexpr index_t IR2 = DstFromSrcReorder{}.Get(I2);
    constexpr index_t IR3 = DstFromSrcReorder{}.Get(I3);

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    constexpr index_t NLoop = ref_desc.GetElementSize() / BlockSize;

    for(index_t iloop = 0; iloop < NLoop; ++iloop)
    {
        index_t is = threadIdx.x + iloop * BlockSize;

        index_t did[4];

        did[0] = is / ref_desc.GetStride(I0);

        is -= did[0] * ref_desc.GetStride(I0);

        did[1] = is / ref_desc.GetStride(I1);

        is -= did[1] * ref_desc.GetStride(I1);

        did[2] = is / ref_desc.GetStride(I2);

        is -= did[2] * ref_desc.GetStride(I2);

        did[3] = is / ref_desc.GetStride(I3);

        const index_t src_index = src_desc.Get1dIndex(did[0], did[1], did[2], did[3]);

        const index_t dst_index = dst_desc.Get1dIndex(did[IR0], did[IR1], did[IR2], did[IR3]);

        f(p_src[src_index], p_dst[dst_index]);
    }

    constexpr bool has_tail = (ref_desc.GetElementSize() > NLoop * BlockSize);

    if(has_tail)
    {
        index_t is = threadIdx.x + NLoop * BlockSize;

        if(is < ref_desc.GetElementSize())
        {
            index_t did[4];

            did[0] = is / ref_desc.GetStride(I0);

            is -= did[0] * ref_desc.GetStride(I0);

            did[1] = is / ref_desc.GetStride(I1);

            is -= did[1] * ref_desc.GetStride(I1);

            did[2] = is / ref_desc.GetStride(I2);

            is -= did[2] * ref_desc.GetStride(I2);

            did[3] = is / ref_desc.GetStride(I3);

            const index_t src_index = src_desc.Get1dIndex(did[0], did[1], did[2], did[3]);

            const index_t dst_index = dst_desc.Get1dIndex(did[IR0], did[IR1], did[IR2], did[IR3]);

            f(p_src[src_index], p_dst[dst_index]);
        }
    }
}

template <index_t BlockSize, class Float, class DstDesc>
__device__ void blockwise_4d_tensor_set_zero(DstDesc, Float* __restrict__ p_dst)
{
    auto f_set_zero = [](Float& v) { v = Float(0); };

    blockwise_4d_tensor_pointwise_operation_unary<BlockSize>(DstDesc{}, p_dst, f_set_zero);
}

template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class DstFromSrcReorder>
__device__ void
blockwise_4d_tensor_copy_reorder_by_get_dst_from_src(SrcDesc,
                                                     const Float* __restrict__ p_src,
                                                     DstDesc,
                                                     Float* __restrict__ p_dst,
                                                     SrcOpLengths,
                                                     DstFromSrcReorder)
{
    auto f_copy = [](const Float& src, Float& dst) { dst = src; };

    blockwise_4d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src<BlockSize>(
        SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, DstFromSrcReorder{}, f_copy);
}

template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class CopyLengths,
          index_t DataPerRead>
struct Blockwise4dTensorCopy1
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    __device__ constexpr Blockwise4dTensorCopy1()
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        static_assert(DataPerRead == 1 ||
                          (SrcDesc{}.GetStride(I3) == 1 && DstDesc{}.GetStride(I3) == 1),
                      "wrong! only support stride3 == 1 if DataPerRead > 1!\n");

        static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                      "wrong! only support DataPerRead == 1, 2 or 4!\n");

        static_assert(SrcDesc{}.GetStride(I2) % DataPerRead == 0 &&
                          DstDesc{}.GetStride(I2) % DataPerRead == 0,
                      "src and dst stride2 should be multiple of DataPerRead to keep alignment");

        // we allow out-of-bound read from src in D3 dimension,
        //   but we need to make sure dst stride2 is big enough,
        //   so that the out-of-bound write won't contaminate next line in dst
        constexpr index_t L3          = CopyLengths{}.Get(I3);
        constexpr index_t read_per_d3 = integer_divide_ceil(L3, DataPerRead);

        static_assert(read_per_d3 * DataPerRead <= DstDesc{}.GetStride(I2),
                      "wrong! out-of-bound write will contaminate next line!\n");
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto src_desc = SrcDesc{};
        constexpr auto dst_desc = DstDesc{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);
        constexpr index_t L3 = CopyLengths{}.Get(I3);

        constexpr index_t read_per_d3 = integer_divide_ceil(L3, DataPerRead);

        constexpr auto ref_desc =
            make_ConstantTensorDescriptor(Sequence<L0, L1, L2, read_per_d3>{});

        constexpr index_t NLoop = ref_desc.GetElementSize() / BlockSize;

        auto f_copy = [&](index_t is) {
            index_t did[4];

            did[0] = is / ref_desc.GetStride(I0);

            is -= did[0] * ref_desc.GetStride(I0);

            did[1] = is / ref_desc.GetStride(I1);

            is -= did[1] * ref_desc.GetStride(I1);

            did[2] = is / ref_desc.GetStride(I2);

            is -= did[2] * ref_desc.GetStride(I2);

            did[3] = is / ref_desc.GetStride(I3);

            const index_t src_index =
                src_desc.Get1dIndex(did[0], did[1], did[2], did[3] * DataPerRead);
            const index_t dst_index =
                dst_desc.Get1dIndex(did[0], did[1], did[2], did[3] * DataPerRead);

            *(reinterpret_cast<vector_t*>(p_dst + dst_index)) =
                *(reinterpret_cast<const vector_t*>(p_src + src_index));
        };

        for(index_t iloop = 0; iloop < NLoop; ++iloop)
        {
            index_t is = threadIdx.x + iloop * BlockSize;

            f_copy(is);
        }

        constexpr bool has_tail = (ref_desc.GetElementSize() > NLoop * BlockSize);

        if(has_tail)
        {
            index_t is = threadIdx.x + NLoop * BlockSize;

            if(is < ref_desc.GetElementSize())
            {
                f_copy(is);
            }
        }
    }
};

template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class DstOpLengths,
          class GlobalLowerPads>
struct BlockwiseChwnTensorCopyPadded
{
    __device__ void Run(const Float* __restrict__ p_src,
                        index_t c_block_data_begin,
                        index_t ho_block_data_begin,
                        index_t wo_block_data_begin,
                        index_t n_block_data_begin,
                        Float* __restrict__ p_dst,
                        index_t h_block_pad_low,
                        index_t w_block_pad_low,
                        index_t h_block_pad_up,
                        index_t w_block_pad_up) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto src_desc = SrcDesc{};
        constexpr auto dst_desc = DstDesc{};
        constexpr auto ref_desc = make_ConstantTensorDescriptor(DstOpLengths{});

        constexpr auto h_global_pad_low = GlobalLowerPads{}.Get(I0);
        constexpr auto w_global_pad_low = GlobalLowerPads{}.Get(I1);

        constexpr index_t NLoop = ref_desc.GetElementSize() / BlockSize;

        const Float* p_src_tmp =
            p_src +
            src_desc.Get1dIndex(c_block_data_begin,
                                (ho_block_data_begin + h_block_pad_low) - h_global_pad_low,
                                (wo_block_data_begin + w_block_pad_low) - w_global_pad_low,
                                n_block_data_begin);

#if 0
        if(get_thread_local_1d_id() == 0)
        {
            print_ConstantTensorDescriptor(src_desc, "src_desc: ");
            print_ConstantTensorDescriptor(dst_desc, "dst_desc: ");
            print_ConstantTensorDescriptor(ref_desc, "ref_desc: ");

            printf("%u %u, \t"
                   "h_global_pad_low %u w_global_pad_low %u \t"
                   "h_block_pad_low %u w_block_pad_low %u h_block_pad_up %u  w_block_pad_up %u \t"
                   "\n",
                   get_block_1d_id(),
                   get_thread_local_1d_id(),
                   h_global_pad_low,
                   w_global_pad_low,
                   h_block_pad_low,
                   w_block_pad_low,
                   h_block_pad_up,
                   w_block_pad_up);
        }
#endif

        for(index_t iloop = 0; iloop < NLoop; ++iloop)
        {
            index_t is = threadIdx.x + iloop * BlockSize;

            index_t did[4];

            did[0] = is / ref_desc.GetStride(I0);

            is -= did[0] * ref_desc.GetStride(I0);

            did[1] = is / ref_desc.GetStride(I1);

            is -= did[1] * ref_desc.GetStride(I1);

            did[2] = is / ref_desc.GetStride(I2);

            is -= did[2] * ref_desc.GetStride(I2);

            did[3] = is / ref_desc.GetStride(I3);

            const index_t bindex = dst_desc.Get1dIndex(did[0], did[1], did[2], did[3]);

            p_dst[bindex] =
                (did[1] < h_block_pad_low || did[1] + h_block_pad_up >= ref_desc.GetLength(I1) ||
                 did[2] < w_block_pad_low || did[2] + w_block_pad_up >= ref_desc.GetLength(I2))
                    ? Float(0)
                    : p_src_tmp[src_desc.Get1dIndex(did[0], did[1], did[2], did[3])];
        }

        constexpr bool has_tail = (ref_desc.GetElementSize() > NLoop * BlockSize);

        if(has_tail)
        {
            index_t is = threadIdx.x + NLoop * BlockSize;

            if(is < ref_desc.GetElementSize())
            {
                index_t did[4];

                did[0] = is / ref_desc.GetStride(I0);

                is -= did[0] * ref_desc.GetStride(I0);

                did[1] = is / ref_desc.GetStride(I1);

                is -= did[1] * ref_desc.GetStride(I1);

                did[2] = is / ref_desc.GetStride(I2);

                is -= did[2] * ref_desc.GetStride(I2);

                did[3] = is / ref_desc.GetStride(I3);

                const index_t bindex = dst_desc.Get1dIndex(did[0], did[1], did[2], did[3]);

                p_dst[bindex] =
                    (did[1] < h_block_pad_low ||
                     did[1] + h_block_pad_up >= ref_desc.GetLength(I1) ||
                     did[2] < w_block_pad_low || did[2] + w_block_pad_up >= ref_desc.GetLength(I2))
                        ? Float(0)
                        : p_src_tmp[src_desc.Get1dIndex(did[0], did[1], did[2], did[3])];
            }
        }
    }
};

// starting point need to be aligned to float4 or float2 or float
// stride3 need to be 1 for both source and destination
template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class CopyLengths,
          class ThreadPerDims,
          index_t DataPerRead>
struct Blockwise4dTensorCopy3
{
    using vector_t = typename vector_type<Float, DataPerRead>::MemoryType;

    index_t mSrcMyThreadOffset;
    index_t mDstMyThreadOffset;

    __device__ Blockwise4dTensorCopy3()
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        static_assert(DataPerRead == 1 ||
                          (SrcDesc{}.GetStride(I3) == 1 && DstDesc{}.GetStride(I3) == 1),
                      "wrong! only support stride3 == 1 if DataPerRead > 1!\n");

        static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                      "wrong! only support DataPerRead == 1, 2 or 4!\n");

        static_assert(
            SrcDesc{}.GetStride(I2) % DataPerRead == 0 &&
                DstDesc{}.GetStride(I2) % DataPerRead == 0,
            "wrong! src and dst stride2 should be multiple of DataPerRead to keep alignment");

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);
        constexpr index_t L3 = CopyLengths{}.Get(I3);

        constexpr index_t thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr index_t thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr index_t thread_per_d2 = ThreadPerDims{}.Get(I2);
        constexpr index_t thread_per_d3 = ThreadPerDims{}.Get(I3);

        // we allow out-of-bound read from src in D3 dimension,
        //   but we need to make sure dst stride is big enough,
        //   so that the out-of-bound write won't contaminate next line in dst
        constexpr index_t nloop_d3 = integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

        static_assert(nloop_d3 * thread_per_d3 * DataPerRead <= DstDesc{}.GetStride(I2),
                      "wrong! out-of-bound write will contaminate next line!\n");

        static_assert(L0 % thread_per_d0 == 0 && L1 % thread_per_d1 == 0 && L2 % thread_per_d2 == 0,
                      "wrong! L0, L1, L2 should be divided evenly!\n");

        static_assert(BlockSize >= thread_per_d0 * thread_per_d1 * thread_per_d2 * thread_per_d3,
                      "wrrong! BlockSize is not big enough for ThreadPerDims!");

        constexpr index_t num_active_thread =
            thread_per_d0 * thread_per_d1 * thread_per_d2 * thread_per_d3;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        const index_t thread_id_d0 =
            get_thread_local_1d_id() / (thread_per_d1 * thread_per_d2 * thread_per_d3);
        index_t itmp = get_thread_local_1d_id() -
                       thread_id_d0 * (thread_per_d1 * thread_per_d2 * thread_per_d3);
        const index_t thread_id_d1 = itmp / (thread_per_d2 * thread_per_d3);
        itmp -= thread_id_d1 * (thread_per_d2 * thread_per_d3);
        const index_t thread_id_d2 = itmp / thread_per_d3;
        const index_t thread_id_d3 = itmp - thread_id_d2 * thread_per_d3;

        mSrcMyThreadOffset = SrcDesc{}.Get1dIndex(
            thread_id_d0, thread_id_d1, thread_id_d2, thread_id_d3 * DataPerRead);
        mDstMyThreadOffset = DstDesc{}.Get1dIndex(
            thread_id_d0, thread_id_d1, thread_id_d2, thread_id_d3 * DataPerRead);
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);
        constexpr index_t L3 = CopyLengths{}.Get(I3);

        constexpr index_t thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr index_t thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr index_t thread_per_d2 = ThreadPerDims{}.Get(I2);
        constexpr index_t thread_per_d3 = ThreadPerDims{}.Get(I3);

        constexpr index_t num_active_thread =
            thread_per_d0 * thread_per_d1 * thread_per_d2 * thread_per_d3;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;
        constexpr index_t nloop_d1 = L1 / thread_per_d1;
        constexpr index_t nloop_d2 = L2 / thread_per_d2;
        constexpr index_t nloop_d3 = integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

#pragma unroll
        for(index_t iloop_d0 = 0; iloop_d0 < nloop_d0; ++iloop_d0)
        {
#pragma unroll
            for(index_t iloop_d1 = 0; iloop_d1 < nloop_d1; ++iloop_d1)
            {
#pragma unroll
                for(index_t iloop_d2 = 0; iloop_d2 < nloop_d2; ++iloop_d2)
                {
#pragma unroll
                    for(index_t iloop_d3 = 0; iloop_d3 < nloop_d3; ++iloop_d3)
                    {
                        const index_t src_offset =
                            SrcDesc{}.Get1dIndex(iloop_d0 * thread_per_d0,
                                                 iloop_d1 * thread_per_d1,
                                                 iloop_d2 * thread_per_d2,
                                                 iloop_d3 * thread_per_d3 * DataPerRead);

                        const index_t dst_offset =
                            DstDesc{}.Get1dIndex(iloop_d0 * thread_per_d0,
                                                 iloop_d1 * thread_per_d1,
                                                 iloop_d2 * thread_per_d2,
                                                 iloop_d3 * thread_per_d3 * DataPerRead);

                        *(reinterpret_cast<vector_t*>(&p_dst[dst_offset + mDstMyThreadOffset])) =
                            *(reinterpret_cast<const vector_t*>(
                                &p_src[src_offset + mSrcMyThreadOffset]));
                    }
                }
            }
        }
    }

    __device__ constexpr index_t GetRegisterClipboardSize() const
    {
        static_assert(is_same<Float, float>::value, "wrong! only support float!\n");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);
        constexpr index_t L3 = CopyLengths{}.Get(I3);

        constexpr index_t thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr index_t thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr index_t thread_per_d2 = ThreadPerDims{}.Get(I2);
        constexpr index_t thread_per_d3 = ThreadPerDims{}.Get(I3);

        constexpr index_t nloop_d0 = L0 / thread_per_d0;
        constexpr index_t nloop_d1 = L1 / thread_per_d1;
        constexpr index_t nloop_d2 = L2 / thread_per_d2;
        constexpr index_t nloop_d3 = integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

        return DataPerRead * nloop_d0 * nloop_d1 * nloop_d2 * nloop_d3;
    }

    __device__ void RunLoadRegisterClipboard(const Float* __restrict__ p_src,
                                             Float* __restrict__ p_clipboard) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);
        constexpr index_t L3 = CopyLengths{}.Get(I3);

        constexpr index_t thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr index_t thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr index_t thread_per_d2 = ThreadPerDims{}.Get(I2);
        constexpr index_t thread_per_d3 = ThreadPerDims{}.Get(I3);

        constexpr index_t num_active_thread =
            thread_per_d0 * thread_per_d1 * thread_per_d2 * thread_per_d3;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;
        constexpr index_t nloop_d1 = L1 / thread_per_d1;
        constexpr index_t nloop_d2 = L2 / thread_per_d2;
        constexpr index_t nloop_d3 = integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

#pragma unroll
        for(index_t iloop_d0 = 0; iloop_d0 < nloop_d0; ++iloop_d0)
        {
#pragma unroll
            for(index_t iloop_d1 = 0; iloop_d1 < nloop_d1; ++iloop_d1)
            {
#pragma unroll
                for(index_t iloop_d2 = 0; iloop_d2 < nloop_d2; ++iloop_d2)
                {
#pragma unroll
                    for(index_t iloop_d3 = 0; iloop_d3 < nloop_d3; ++iloop_d3)
                    {
                        const index_t src_offset =
                            SrcDesc{}.Get1dIndex(iloop_d0 * thread_per_d0,
                                                 iloop_d1 * thread_per_d1,
                                                 iloop_d2 * thread_per_d2,
                                                 iloop_d3 * thread_per_d3 * DataPerRead);

                        const index_t dst_offset =
                            DstDesc{}.Get1dIndex(iloop_d0 * thread_per_d0,
                                                 iloop_d1 * thread_per_d1,
                                                 iloop_d2 * thread_per_d2,
                                                 iloop_d3 * thread_per_d3 * DataPerRead);

                        *(reinterpret_cast<vector_t*>(&p_clipboard[dst_offset])) =
                            *(reinterpret_cast<const vector_t*>(
                                &p_src[src_offset + mSrcMyThreadOffset]));
                    }
                }
            }
        }
    }

    __device__ void RunStoreRegisterClipboard(const Float* __restrict__ p_clipboard,
                                              Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr index_t L0 = CopyLengths{}.Get(I0);
        constexpr index_t L1 = CopyLengths{}.Get(I1);
        constexpr index_t L2 = CopyLengths{}.Get(I2);
        constexpr index_t L3 = CopyLengths{}.Get(I3);

        constexpr index_t thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr index_t thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr index_t thread_per_d2 = ThreadPerDims{}.Get(I2);
        constexpr index_t thread_per_d3 = ThreadPerDims{}.Get(I3);

        constexpr index_t num_active_thread =
            thread_per_d0 * thread_per_d1 * thread_per_d2 * thread_per_d3;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr index_t nloop_d0 = L0 / thread_per_d0;
        constexpr index_t nloop_d1 = L1 / thread_per_d1;
        constexpr index_t nloop_d2 = L2 / thread_per_d2;
        constexpr index_t nloop_d3 = integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

#pragma unroll
        for(index_t iloop_d0 = 0; iloop_d0 < nloop_d0; ++iloop_d0)
        {
#pragma unroll
            for(index_t iloop_d1 = 0; iloop_d1 < nloop_d1; ++iloop_d1)
            {
#pragma unroll
                for(index_t iloop_d2 = 0; iloop_d2 < nloop_d2; ++iloop_d2)
                {
#pragma unroll
                    for(index_t iloop_d3 = 0; iloop_d3 < nloop_d3; ++iloop_d3)
                    {
                        const index_t src_offset =
                            SrcDesc{}.Get1dIndex(iloop_d0 * thread_per_d0,
                                                 iloop_d1 * thread_per_d1,
                                                 iloop_d2 * thread_per_d2,
                                                 iloop_d3 * thread_per_d3 * DataPerRead);

                        const index_t dst_offset =
                            DstDesc{}.Get1dIndex(iloop_d0 * thread_per_d0,
                                                 iloop_d1 * thread_per_d1,
                                                 iloop_d2 * thread_per_d2,
                                                 iloop_d3 * thread_per_d3 * DataPerRead);

                        *(reinterpret_cast<vector_t*>(&p_dst[dst_offset + mDstMyThreadOffset])) =
                            *(reinterpret_cast<const vector_t*>(&p_clipboard[src_offset]));
                    }
                }
            }
        }
    }
};
