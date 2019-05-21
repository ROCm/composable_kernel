#pragma once
#include "threadwise_tensor_slice_op.hip.hpp"

// slice a merged tensor, reorder and copy it into a normal tensor
// src: a merged tensor,
// dst: a normal tensor
template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SliceLengths,
          class SubLengths,
          class ClusterLengths,
          class ThreadClusterArrangeOrder,
          class SrcAccessOrder,
          class DstAccessOrder>
struct BlockwiseTensorSliceCopy_generic_v1
{
    static constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    index_t mSrcMyThreadOffset;
    index_t mDstMyThreadOffset;

    __device__ BlockwiseTensorSliceCopy_generic_v1(Array<index_t, nDim> src_block_multi_offset,
                                                   Array<index_t, nDim> dst_block_multi_offset)
    {
        // check NDim consistent
        static_assert(SrcDesc::GetNumOfDimension() == DstDesc::GetNumOfDimension(), "wrong");

        constexpr auto thread_cluster_desc = make_packed_ConstantTensorDescriptor(
            ClusterLengths{}.ReorderGivenNew2Old(ThreadClusterArrangeOrder{}));

        // BlockSize
        static_assert(BlockSize == thread_cluster_desc.GetElementSize(), "wrong! BlockSize");

        // divide work
        static_for<0, nDim, 1>{}([&](auto IDim) {
            static_assert(SliceLengths{}.Get(IDim) % SubLenghs{}.Get(IDim) == 0,
                          "wrong! cannot evenly divide sliced tensor into sub-tensor");
        });

        constexpr auto thread_work_desc =
            make_packed_ConstantTensorDescriptor(SliceLengths{} / SliceSubLengths{});

        static_for<0, nDim, 1>{}([&](auto IDim) {
            static_assert(thread_work_desc.GetLength(IDim) % thread_cluster_desc.Get(IDim) == 0,
                          "wrong! cannot evenly divide work to cluster");
        });

        // only support SubLengths.Get() == 1 on merged dimension, for now
        static_for<0, nDim, 1>{}([&](auto IDim) {
            static_if<(SrcDesc::ContainMultipleOriginalDimensions(IDim) ||
                       DstDesc::ContainMultipleOriginalDimensions(IDim))>{}([&](auto fwd) {
                static_assert(fwd(SubLengths{}).Get(IDim) == 1,
                              "wrong! Sub-Lengths on merged dimension should be 1");
            });
        });

        // calculate mSrcMyThreadOffset, mDstMyThreadOffset
        const auto thread_cluster_multi_id =
            thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        const auto data_cluster_multi_id =
            reorder_array_given_old2new(thread_cluster_multi_id, ThreadClusterArrangeOrder{});

        const auto thread_data_multi_offset = data_cluster_multi_id * SubLengths{};

        mSrcMythreadOffset =
            SrcDesc::GetOffsetFromMultiIndex(src_block_multi_offset + thread_data_multi_offset);
        mSrcMythreadOffset =
            DstDesc::GetOffsetFromMultiIndex(dst_block_multi_offset + thread_data_multi_offset);
    }

    __device__ static constexpr index_t GetRegisterClipboardSize()
    {
        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * ClusterLengths{});

        constexpr auto thread_tensor_desc =
            make_packed_ConstantTensorDescriptor(SubLengths{} * repeat_lengths);

        return thread_tensor_desc.GetElementSpaceSize();
    }

    __device__ void RunLoadRegisterClipboard(const Float* __restrict__ p_src,
                                             Float* __restrict__ p_clipboard) const
    {
        constexpr auto thread_sub_tensor_lengths = SubLengths{};

        constexpr auto data_per_cluster_per_dims = thread_sub_tensor_lengths * ClusterLengths{};

        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * ClusterLengths{});

        constexpr auto thread_tensor_desc =
            make_packed_ConstantTensorDescriptor(thread_sub_tensor_lengths * repeat_lengths);

        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id_) {
            constexpr auto repeat_multi_id = decltype(repeat_multi_id_){};

            constexpr auto src_data_multi_offset = repeat_multi_id * data_per_cluster_per_dims;

            constexpr auto clipboard_data_multi_offset =
                repeat_multi_id * thread_sub_tensor_lengths;

            constexpr index_t src_offset = SrcDesc{}.GetOffsetFromMultiIndex(src_data_multi_id);
            constexpr index_t clipboard_offset =
                thread_tensor_desc.GetOffsetFromMultiIndex(clipboard_data_multi_id);

            threadwise_tensor_slice_copy_generic(SrcDesc{},
                                                 p_src + src_offset + mSrcMyThreadOffset,
                                                 thread_tensor_desc,
                                                 zero_array<index_t, nDim>{},
                                                 thread_tensor_desc,
                                                 p_clipboard + clipboard_offset,
                                                 zero_array<index_t, nDim>{},
                                                 thread_sub_tensor_lengths,
                                                 SrcAccessOrder{});
        });
    }

    __device__ void RunStoreRegisterClipboard(const Float* __restrict__ p_clipboard,
                                              Float* __restrict__ p_dst) const
    {
        constexpr auto thread_sub_tensor_lengths = SubLengths{};

        constexpr auto data_per_cluster_per_dims = thread_sub_tensor_lengths * ClusterLengths{};

        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * ClusterLengths{});

        constexpr auto thread_tensor_desc =
            make_packed_ConstantTensorDescriptor(thread_sub_tensor_lengths * repeat_lengths);

        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id_) {
            constexpr auto repeat_multi_id = decltype(repeat_multi_id_){};

            constexpr auto clipboard_data_multi_offset =
                repeat_multi_id * thread_sub_tensor_lengths;

            constexpr auto dst_data_multi_offset = repeat_multi_id * data_per_cluster_per_dims;

            constexpr index_t clipboard_offset =
                thread_tensor_desc.GetOffsetFromMultiIndex(clipboard_data_multi_offset);

            constexpr index_t dst_offset = DstDesc{}.GetOffsetFromMultiIndex(dst_data_multi_offset);

            threadwise_tensor_slice_copy_generic(thread_tensor_desc,
                                                 p_clipboard + clipboard_offset,
                                                 zero_array<index_t, nDim>{},
                                                 DstDesc{},
                                                 p_dst + dst_offset + mDstMyThreadOffset,
                                                 zero_array<index_t, nDim>{},
                                                 thread_sub_tensor_lengths,
                                                 DstAccessOrder{});
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
            Float p_clipboard[GetRegisterClipboardSize()];

            RunLoadRegisterClipboard(p_src, p_clipboard);
            RunStoreRegisterClipboard(p_clipboard, p_dst);
    }
    };
