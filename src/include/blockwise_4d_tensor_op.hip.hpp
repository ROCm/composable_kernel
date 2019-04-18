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
        index_t is = get_thread_local_1d_id() + NLoop * BlockSize;

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
          class MapDst2Src,
          class F>
__device__ void blockwise_4d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src(
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
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr index_t IR0 = MapDst2Src{}.Get(I0);
    constexpr index_t IR1 = MapDst2Src{}.Get(I1);
    constexpr index_t IR2 = MapDst2Src{}.Get(I2);
    constexpr index_t IR3 = MapDst2Src{}.Get(I3);

    constexpr auto src_desc = SrcDesc{};
    constexpr auto dst_desc = DstDesc{};
    constexpr auto ref_desc = make_ConstantTensorDescriptor(SrcOpLengths{});

    constexpr index_t NLoop = ref_desc.GetElementSize() / BlockSize;

    for(index_t iloop = 0; iloop < NLoop; ++iloop)
    {
        index_t is = get_thread_local_1d_id() + iloop * BlockSize;

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
        index_t is = get_thread_local_1d_id() + NLoop * BlockSize;

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
          class MapDst2Src>
__device__ void
blockwise_4d_tensor_copy_reorder_by_get_dst_from_src(SrcDesc,
                                                     const Float* __restrict__ p_src,
                                                     DstDesc,
                                                     Float* __restrict__ p_dst,
                                                     SrcOpLengths,
                                                     MapDst2Src)
{
    auto f_copy = [](const Float& src, Float& dst) { dst = src; };

    blockwise_4d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src<BlockSize>(
        SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, MapDst2Src{}, f_copy);
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
        constexpr index_t read_per_d3 = mod_conv::integer_divide_ceil(L3, DataPerRead);

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

        constexpr index_t read_per_d3 = mod_conv::integer_divide_ceil(L3, DataPerRead);

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
            p_src + src_desc.Get1dIndex(c_block_data_begin,
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
            index_t is = get_thread_local_1d_id() + iloop * BlockSize;

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
            index_t is = get_thread_local_1d_id() + NLoop * BlockSize;

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
        constexpr index_t nloop_d3 = mod_conv::integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

        static_assert(nloop_d3 * thread_per_d3 * DataPerRead <= DstDesc{}.GetStride(I2),
                      "wrong! out-of-bound write will contaminate next line!\n");

        static_assert(L0 % thread_per_d0 == 0 && L1 % thread_per_d1 == 0 && L2 % thread_per_d2 == 0,
                      "wrong! L0, L1, L2 should be divided evenly!\n");

        static_assert(BlockSize >= thread_per_d0 * thread_per_d1 * thread_per_d2 * thread_per_d3,
                      "wrrong! BlockSize is not big enough for ThreadPerDims!");

        constexpr index_t num_active_thread =
            accumulate_on_sequence(ThreadPerDims{}, mod_conv::multiplies<index_t>{}, Number<1>{});

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor(ThreadPerDims{});
        const auto thread_multi_id = thread_cluster_desc.GetMultiIndex(get_thread_local_1d_id());

        mSrcMyThreadOffset = SrcDesc{}.Get1dIndex(thread_multi_id[0],
                                                  thread_multi_id[1],
                                                  thread_multi_id[2],
                                                  thread_multi_id[3] * DataPerRead);

        mDstMyThreadOffset = DstDesc{}.Get1dIndex(thread_multi_id[0],
                                                  thread_multi_id[1],
                                                  thread_multi_id[2],
                                                  thread_multi_id[3] * DataPerRead);
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
        constexpr index_t nloop_d3 = mod_conv::integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

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
        constexpr index_t nloop_d3 = mod_conv::integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

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
        constexpr index_t nloop_d3 = mod_conv::integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

        constexpr auto clipboard_desc = make_ConstantTensorDescriptor(
            Sequence<nloop_d0, nloop_d1, nloop_d2, nloop_d3 * DataPerRead>{});

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

                        const index_t clipboard_offset = clipboard_desc.Get1dIndex(
                            iloop_d0, iloop_d1, iloop_d2, iloop_d3 * DataPerRead);

                        *(reinterpret_cast<vector_t*>(&p_clipboard[clipboard_offset])) =
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
        constexpr index_t nloop_d3 = mod_conv::integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

        constexpr auto clipboard_desc = make_ConstantTensorDescriptor(
            Sequence<nloop_d0, nloop_d1, nloop_d2, nloop_d3 * DataPerRead>{});

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
                        const index_t clipboard_offset = clipboard_desc.Get1dIndex(
                            iloop_d0, iloop_d1, iloop_d2, iloop_d3 * DataPerRead);

                        const index_t dst_offset =
                            DstDesc{}.Get1dIndex(iloop_d0 * thread_per_d0,
                                                 iloop_d1 * thread_per_d1,
                                                 iloop_d2 * thread_per_d2,
                                                 iloop_d3 * thread_per_d3 * DataPerRead);

                        *(reinterpret_cast<vector_t*>(&p_dst[dst_offset + mDstMyThreadOffset])) =
                            *(reinterpret_cast<const vector_t*>(&p_clipboard[clipboard_offset]));
                    }
                }
            }
        }
    }
};

template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcOpLengths,
          class MapDst2Src>
struct Blockwise4dTensorCopyReorder1
{
    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        auto f_copy = [](const Float& src, Float& dst) { dst = src; };

        blockwise_4d_tensor_pointwise_operation_binary_reorder_by_get_dst_from_src<BlockSize>(
            SrcDesc{}, p_src, DstDesc{}, p_dst, SrcOpLengths{}, MapDst2Src{}, f_copy);
    }
};

#if 1
template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SrcLengths,
          class SrcSubLengths,
          class SrcClusterLengths,
          class MapDst2Src,
          class MapThreadCluster2SrcCluster,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite>
struct Blockwise4dTensorCopyReorder3
{
    static constexpr index_t nDim = SrcLengths::GetSize();

    index_t mSrcMyThreadOffset;
    index_t mDstMyThreadOffset;

    __device__ Blockwise4dTensorCopyReorder3()
    {
        constexpr auto src_desc = SrcDesc{};
        constexpr auto dst_desc = DstDesc{};

        constexpr auto src_lengths = SrcLengths{};

        constexpr auto map_dst2src = MapDst2Src{};

        constexpr auto src_sub_lengths = SrcSubLengths{};
        constexpr auto dst_sub_lengths = src_sub_lengths.ReorderGivenNew2Old(map_dst2src);

        constexpr auto map_thread_cluster_2_src_cluster = MapThreadCluster2SrcCluster{};

        constexpr auto src_cluster_lengths = SrcClusterLengths{};
        constexpr auto thread_cluster_lengths =
            src_cluster_lengths.ReorderGivenNew2Old(map_thread_cluster_2_src_cluster);

        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor(thread_cluster_lengths);

        // sanity check: data type
        static_assert(is_same<Float, float>::value, "wrong! only support float for now!\n");

        // sanity check: nDim
        static_assert(SrcDesc::GetDimension() == nDim && DstDesc::GetDimension() == nDim &&
                          SrcLengths::GetSize() == nDim && SrcSubLengths::GetSize() == nDim &&
                          SrcClusterLengths::GetSize() == nDim && MapDst2Src::GetSize() == nDim &&
                          MapThreadCluster2SrcCluster::GetSize() == nDim,
                      "wrong! nDim is not consistent\n");

        // sanity check: BlockSize
        constexpr index_t num_active_thread = thread_cluster_desc.GetElementSize();

        static_assert(BlockSize >= num_active_thread,
                      "wrong! BlockSize is not big enough for ThreadPerDims!");

        // sanity check: work division
        static_for<0, nDim, 1>{}([](auto IDim) {
            constexpr auto I                  = decltype(IDim){};
            constexpr index_t src_len         = src_lengths.Get(I);
            constexpr index_t src_sub_len     = src_sub_lengths.Get(I);
            constexpr index_t src_cluster_len = src_cluster_lengths.Get(I);
            static_assert(src_len % (src_sub_len * src_cluster_len) == 0,
                          "wrong! cannot evenly divide Src tensor lengths");
        });

        // sanity check: src read
        static_assert(SrcDataPerRead == 1 || SrcDataPerRead == 2 || SrcDataPerRead == 4,
                      "wrong! only support SrcDataPerRead == 1, 2 or 4!\n");

        static_assert(SrcDataPerRead == 1 || src_desc.GetStride(Number<nDim - 1>{}) == 1,
                      "wrong! only support src.stride(nDim-1) == 1 if SrcDataPerRead > 1!\n");

        static_assert(src_sub_lengths.Get(Number<nDim - 1>{}) % SrcDataPerRead == 0,
                      "wrong! src_sub_lengths[nDim-1] % SrcDataPerRead != 0\n");

        static_assert(src_desc.GetStride(Number<nDim - 2>{}) % SrcDataPerRead == 0,
                      "wrong! should satisfy src_desc.stride(nDim-2) % SrcDataPerRead == 0, to "
                      "keep alignment");

        // sanity check: dst write
        static_assert(DstDataPerWrite == 1 || DstDataPerWrite == 2 || DstDataPerWrite == 4,
                      "wrong! only support DstDataPerWrite == 1, 2 or 4!\n");

        static_assert(DstDataPerWrite == 1 || dst_desc.GetStride(Number<nDim - 1>{}) == 1,
                      "wrong! only support dst.stride(nDim-1) == 1 if DstDataPerWrite > 1!\n");

        static_assert(dst_sub_lengths.Get(Number<nDim - 1>{}) % DstDataPerWrite == 0,
                      "wrong! dst_sub_lengths[nDim-1] % DstDataPerWrite != 0\n");

        static_assert(dst_desc.GetStride(Number<nDim - 2>{}) % DstDataPerWrite == 0,
                      "wrong! should satisfy dst_desc.stride(nDim-2) % DstDataPerWrite == 0, to "
                      "keep alignment");

        // start dividing work
        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        const auto thread_multi_id = thread_cluster_desc.GetMultiIndex(get_thread_local_1d_id());

        // compiler: thread_multi_id, src_data_multi_id, dst_data_multi_id, will use separate
        // regsiters, or only one copy???
        auto src_data_multi_id =
            reorder_array_given_old2new(thread_multi_id, map_thread_cluster_2_src_cluster);

        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr auto I    = decltype(IDim){};
            constexpr index_t i = I.Get();
            // compiler: will it really compute index here, or be associated with Get1dIndex and
            // optimized away???
            src_data_multi_id[i] *= src_sub_lengths.Get(I);
        });

        // compiler: will it really compute index here, or be associated with Get1dIndex and
        // optimized away???
        const auto dst_data_multi_id = reorder_array_given_new2old(src_data_multi_id, map_dst2src);

        mSrcMyThreadOffset = src_desc.Get1dIndex(src_data_multi_id);
        mDstMyThreadOffset = dst_desc.Get1dIndex(dst_data_multi_id);

#if 0
        if(get_block_1d_id() == 0)
        {
            printf("tid %5u, "
                   "thread_multi_id %5u %5u %5u %5u, "
                   "src_data_multi_id %5u %5u %5u %5u, "
                   "dst_data_multi_id %5u %5u %5u %5u, "
                   "mSrcMyThreadOffset %u, mDstMyThreadOffset %u\n",
                   get_thread_local_1d_id(),
                   thread_multi_id[0],
                   thread_multi_id[1],
                   thread_multi_id[2],
                   thread_multi_id[3],
                   src_data_multi_id[0],
                   src_data_multi_id[1],
                   src_data_multi_id[2],
                   src_data_multi_id[3],
                   dst_data_multi_id[0],
                   dst_data_multi_id[1],
                   dst_data_multi_id[2],
                   dst_data_multi_id[3],
                   mSrcMyThreadOffset,
                   mDstMyThreadOffset);
        }
#endif
    }

    __device__ static constexpr index_t GetRegisterClipboardSize()
    {
        constexpr auto thread_sub_tensor_lengths = SrcSubLengths{};

        constexpr auto src_data_per_cluster_per_dims = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, SrcClusterLengths{});

        constexpr auto cluster_per_dims =
            transform_sequences(mod_conv::integer_divide_ceiler<index_t>{},
                                SrcLengths{},
                                src_data_per_cluster_per_dims);

        constexpr auto thread_tensor_lengths = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, cluster_per_dims);

        constexpr auto thread_tensor_desc = make_ConstantTensorDescriptor(thread_tensor_lengths);

        return thread_tensor_desc.GetElementSpace();
    }

    __device__ void RunLoadRegisterClipboard(const Float* __restrict__ p_src,
                                             Float* __restrict__ p_clipboard) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto thread_sub_tensor_lengths = SrcSubLengths{};

        constexpr auto src_data_per_cluster_per_dims = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, SrcClusterLengths{});

        constexpr auto cluster_per_dims =
            transform_sequences(mod_conv::integer_divide_ceiler<index_t>{},
                                SrcLengths{},
                                src_data_per_cluster_per_dims);

        constexpr auto thread_tensor_lengths = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, cluster_per_dims);

        constexpr auto thread_tensor_desc = make_ConstantTensorDescriptor(thread_tensor_lengths);

        constexpr auto thread_sub_tensor_desc =
            make_ConstantTensorDescriptor(SrcClusterLengths{}, thread_tensor_desc.GetStrides());

        for(index_t icluster_d0 = 0; icluster_d0 < cluster_per_dims.Get(I0); ++icluster_d0)
        {
            for(index_t icluster_d1 = 0; icluster_d1 < cluster_per_dims.Get(I1); ++icluster_d1)
            {
                for(index_t icluster_d2 = 0; icluster_d2 < cluster_per_dims.Get(I2); ++icluster_d2)
                {
                    for(index_t icluster_d3 = 0; icluster_d3 < cluster_per_dims.Get(I3);
                        ++icluster_d3)
                    {
                        const index_t src_offset = SrcDesc{}.Get1dIndex(
                            icluster_d0 * src_data_per_cluster_per_dims.Get(I0),
                            icluster_d1 * src_data_per_cluster_per_dims.Get(I1),
                            icluster_d2 * src_data_per_cluster_per_dims.Get(I2),
                            icluster_d3 * src_data_per_cluster_per_dims.Get(I3));

                        const index_t clipboard_offset = thread_tensor_desc.Get1dIndex(
                            icluster_d0 * thread_sub_tensor_lengths.Get(I0),
                            icluster_d1 * thread_sub_tensor_lengths.Get(I1),
                            icluster_d2 * thread_sub_tensor_lengths.Get(I2),
                            icluster_d3 * thread_sub_tensor_lengths.Get(I3));

                        threadwise_4d_tensor_copy_v2(SrcDesc{},
                                                     p_src + src_offset + mSrcMyThreadOffset,
                                                     thread_tensor_desc,
                                                     p_clipboard + clipboard_offset,
                                                     thread_sub_tensor_lengths,
                                                     Number<SrcDataPerRead>{});
                    }
                }
            }
        }

#if 0
        if(get_block_1d_id() == 0)
        {
            printf("tid %5u, "
                   "data: %f %f %f %f %f %f %f %f\n",
                   get_thread_local_1d_id(),
                   p_clipboard[0],
                   p_clipboard[1],
                   p_clipboard[2],
                   p_clipboard[3],
                   p_clipboard[4],
                   p_clipboard[5],
                   p_clipboard[6],
                   p_clipboard[7]);
        }
#endif
    }

    __device__ void RunStoreRegisterClipboard(const Float* __restrict__ p_clipboard,
                                              Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto thread_sub_tensor_lengths = SrcSubLengths{};

        constexpr auto src_data_per_cluster_per_dims = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, SrcClusterLengths{});

        constexpr auto cluster_per_dims =
            transform_sequences(mod_conv::integer_divide_ceiler<index_t>{},
                                SrcLengths{},
                                src_data_per_cluster_per_dims);

        constexpr auto thread_tensor_lengths = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, cluster_per_dims);

        constexpr auto thread_tensor_desc = make_ConstantTensorDescriptor(thread_tensor_lengths);

        constexpr auto thread_sub_tensor_desc =
            make_ConstantTensorDescriptor(SrcClusterLengths{}, thread_tensor_desc.GetStrides());

        for(index_t icluster_d0 = 0; icluster_d0 < cluster_per_dims.Get(I0); ++icluster_d0)
        {
            for(index_t icluster_d1 = 0; icluster_d1 < cluster_per_dims.Get(I1); ++icluster_d1)
            {
                for(index_t icluster_d2 = 0; icluster_d2 < cluster_per_dims.Get(I2); ++icluster_d2)
                {
                    for(index_t icluster_d3 = 0; icluster_d3 < cluster_per_dims.Get(I3);
                        ++icluster_d3)
                    {
                        const index_t clipboard_offset = thread_tensor_desc.Get1dIndex(
                            icluster_d0 * thread_sub_tensor_lengths.Get(I0),
                            icluster_d1 * thread_sub_tensor_lengths.Get(I1),
                            icluster_d2 * thread_sub_tensor_lengths.Get(I2),
                            icluster_d3 * thread_sub_tensor_lengths.Get(I3));

                        const auto dst_multi_id = reorder_array_given_new2old(
                            Array<index_t, nDim>{
                                icluster_d0 * src_data_per_cluster_per_dims.Get(I0),
                                icluster_d1 * src_data_per_cluster_per_dims.Get(I1),
                                icluster_d2 * src_data_per_cluster_per_dims.Get(I2),
                                icluster_d3 * src_data_per_cluster_per_dims.Get(I3)},
                            MapDst2Src{});

                        const index_t dst_offset = DstDesc{}.Get1dIndex(dst_multi_id);

#if 0
                        if(get_block_1d_id() == 0)
                        {
                            printf("tid %5u, "
                                    "clipboard_offsetm %5u, dst_offset %5u\n",
                            get_thread_local_1d_id(),
                            clipboard_offset,
                            dst_offset);
                        }
#endif

#if 1
                        threadwise_4d_tensor_copy_reorder_by_get_dst_from_src(
                            thread_tensor_desc,
                            p_clipboard + clipboard_offset,
                            DstDesc{},
                            p_dst + dst_offset + mDstMyThreadOffset,
                            thread_sub_tensor_lengths,
                            MapDst2Src{});
#endif
                    }
                }
            }
        }

#if 0
        if(get_block_1d_id() == 0)
        {
            printf("tid %5u, "
                   "data: %f %f %f %f %f %f %f %f\n",
                   get_thread_local_1d_id(),
                   p_clipboard[0],
                   p_clipboard[1],
                   p_clipboard[2],
                   p_clipboard[3],
                   p_clipboard[4],
                   p_clipboard[5],
                   p_clipboard[6],
                   p_clipboard[7]);
        }
#endif
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        Float p_clipboard[GetRegisterClipboardSize()];

        RunLoadRegisterClipboard(p_src, p_clipboard);
        RunStoreRegisterClipboard(p_clipboard, p_dst);
    }
};
#endif
