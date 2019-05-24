#pragma once
#include "threadwise_tensor_slice_op.hip.hpp"

// slice a (normal or merged) tensor, reorder and copy it into another (normal or merged) tensor
template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SliceLengths,
          class SubLengths,
          class DataClusterLengths,
          class ThreadClusterArrangeOrder,
          class SrcAccessOrder,
          class DstAccessOrder,
          index_t SrcDataPerRead,
          index_t DstDataPerRead>
struct BlockwiseGenericTensorSliceCopy_v1
{
    static constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    index_t mSrcMyThreadOffset;
    index_t mDstMyThreadOffset;

    __device__
    BlockwiseGenericTensorSliceCopy_v1(Array<index_t, nDim> src_block_data_multi_id_begin,
                                       Array<index_t, nDim> dst_block_data_multi_id_begin)
    {
        // check NDim consistent
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::GetSize() &&
                          nDim == SubLengths::GetSize() && nDim == DataClusterLengths::GetSize() &&
                          nDim == ThreadClusterArrangeOrder::GetSize() &&
                          nDim == SrcAccessOrder::GetSize() && nDim == DstAccessOrder::GetSize(),
                      "wrong");

        // check
        static_assert(is_valid_sequence_map<ThreadClusterArrangeOrder>::value &&
                          is_valid_sequence_map<SrcAccessOrder>::value &&
                          is_valid_sequence_map<DstAccessOrder>::value,
                      "wrong!");

        // thread cluster
        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor_default_rank_packed(
            DataClusterLengths{}.ReorderGivenNew2Old(ThreadClusterArrangeOrder{}));

        // BlockSize
        static_assert(BlockSize == thread_cluster_desc.GetElementSize(), "wrong! BlockSize");

        // divide work
        constexpr auto data_per_cluster_per_dims = SubLengths{} * DataClusterLengths{};

        static_for<0, nDim, 1>{}([&](auto IDim_) {
            constexpr auto IDim = decltype(IDim_){};

            static_assert(SliceLengths::Get(IDim) % SubLengths::Get(IDim) == 0,
                          "wrong! cannot evenly divide sliced tensor into sub-tensor");

            static_assert(SliceLengths::Get(IDim) % data_per_cluster_per_dims.Get(IDim) == 0,
                          "wrong! cannot evenly divide sliced tensor into cluster");
        });

        constexpr auto repeat_lengths = SliceLengths{} / data_per_cluster_per_dims;

        // for now, only support SubLengths.Get() == 1 on a merged dimension that is merge from
        // multiple dimensions
        static_for<0, nDim, 1>{}([&](auto IDim_) {
            constexpr auto IDim = decltype(IDim_){};

            static_assert(SubLengths::Get(IDim) == 1 ||
                              (!SrcDesc::ContainMultipleOriginalDimensions(IDim) &&
                               !DstDesc::ContainMultipleOriginalDimensions(IDim)),
                          "wrong! only surpport Sub-Length == 1 on a merged dimension");
        });

        // calculate mSrcMyThreadOffset, mDstMyThreadOffset
        const auto thread_cluster_multi_id =
            thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        const auto data_cluster_multi_id =
            reorder_array_given_old2new(thread_cluster_multi_id, ThreadClusterArrangeOrder{});

        const auto thread_data_multi_id_begin = data_cluster_multi_id * SubLengths{};

        mSrcMyThreadOffset = SrcDesc::GetOffsetFromMultiIndex(src_block_data_multi_id_begin +
                                                              thread_data_multi_id_begin);

        mDstMyThreadOffset = DstDesc::GetOffsetFromMultiIndex(dst_block_data_multi_id_begin +
                                                              thread_data_multi_id_begin);
#if 0
        {
            printf("id %5u %5u: "
                   "src_block_data_multi_id_begin: %u %u %u %u, "
                   "thread_cluster_multi_id: %u %u %u %u, "
                   "data_cluster_multi_id: %u %u %u %u, "
                   "thread_data_multi_id_begin: %u %u %u %u, "
                   "mSrcMyThreadOffset %u, mDstMyThreadOffset %u \n",
                   get_block_1d_id(),
                   get_thread_local_1d_id(),
                   src_block_data_multi_id_begin[0],
                   src_block_data_multi_id_begin[1],
                   src_block_data_multi_id_begin[2],
                   src_block_data_multi_id_begin[3],
                   thread_cluster_multi_id[0],
                   thread_cluster_multi_id[1],
                   thread_cluster_multi_id[2],
                   thread_cluster_multi_id[3],
                   data_cluster_multi_id[0],
                   data_cluster_multi_id[1],
                   data_cluster_multi_id[2],
                   data_cluster_multi_id[3],
                   thread_data_multi_id_begin[0],
                   thread_data_multi_id_begin[1],
                   thread_data_multi_id_begin[2],
                   thread_data_multi_id_begin[3],
                   mSrcMyThreadOffset,
                   mDstMyThreadOffset);
        }
#endif
    }

    __device__ static constexpr index_t GetRegisterClipboardSize()
    {
        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * DataClusterLengths{});

        constexpr auto thread_tensor_desc =
            make_ConstantTensorDescriptor_default_rank_packed(SubLengths{} * repeat_lengths);

        return thread_tensor_desc.GetElementSpace();
    }

    __device__ void RunLoadRegisterClipboard(const Float* __restrict__ p_src,
                                             Float* __restrict__ p_clipboard) const
    {
        constexpr auto thread_sub_tensor_lengths = SubLengths{};

        constexpr auto data_per_cluster_per_dims = thread_sub_tensor_lengths * DataClusterLengths{};

        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * DataClusterLengths{});

        constexpr auto thread_tensor_desc = make_ConstantTensorDescriptor_default_rank_packed(
            thread_sub_tensor_lengths * repeat_lengths);

        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id_) {
            constexpr auto repeat_multi_id = sequence2array(decltype(repeat_multi_id_){});

            const auto src_thread_data_multi_id_begin =
                repeat_multi_id * data_per_cluster_per_dims; // cannot not constexpr, why?

            const auto clipboard_data_multi_id_begin =
                repeat_multi_id * thread_sub_tensor_lengths; // cannot not constexpr, why?

            const index_t src_offset = SrcDesc{}.GetOffsetFromMultiIndex(
                src_thread_data_multi_id_begin); // cannot not constexpr, why?

            const index_t clipboard_offset = thread_tensor_desc.GetOffsetFromMultiIndex(
                clipboard_data_multi_id_begin); // cannot not constexpr, why?

            threadwise_generic_tensor_slice_copy(SrcDesc{},
                                                 p_src + src_offset + mSrcMyThreadOffset,
                                                 make_zero_array<index_t, nDim>(),
                                                 thread_tensor_desc,
                                                 p_clipboard + clipboard_offset,
                                                 make_zero_array<index_t, nDim>(),
                                                 thread_sub_tensor_lengths,
                                                 SrcAccessOrder{});
        });
    }

    __device__ void RunStoreRegisterClipboard(const Float* __restrict__ p_clipboard,
                                              Float* __restrict__ p_dst) const
    {
        constexpr auto thread_sub_tensor_lengths = SubLengths{};

        constexpr auto data_per_cluster_per_dims = thread_sub_tensor_lengths * DataClusterLengths{};

        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * DataClusterLengths{});

        constexpr auto thread_tensor_desc = make_ConstantTensorDescriptor_default_rank_packed(
            thread_sub_tensor_lengths * repeat_lengths);

        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id_) {
            constexpr auto repeat_multi_id = sequence2array(decltype(repeat_multi_id_){});

            const auto clipboard_data_multi_id_begin =
                repeat_multi_id * thread_sub_tensor_lengths; // cannot not constexpr, why?

            const auto dst_data_multi_id_begin =
                repeat_multi_id * data_per_cluster_per_dims; // cannot not constexpr, why?

            const index_t clipboard_offset = thread_tensor_desc.GetOffsetFromMultiIndex(
                clipboard_data_multi_id_begin); // cannot not constexpr, why?

            const index_t dst_offset = DstDesc{}.GetOffsetFromMultiIndex(
                dst_data_multi_id_begin); // cannot not constexpr, why?

            threadwise_generic_tensor_slice_copy(thread_tensor_desc,
                                                 p_clipboard + clipboard_offset,
                                                 make_zero_array<index_t, nDim>(),
                                                 DstDesc{},
                                                 p_dst + dst_offset + mDstMyThreadOffset,
                                                 make_zero_array<index_t, nDim>(),
                                                 thread_sub_tensor_lengths,
                                                 DstAccessOrder{});
        });
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        Float p_clipboard[GetRegisterClipboardSize()];

        RunLoadRegisterClipboard(p_src, p_clipboard);
        RunStoreRegisterClipboard(p_clipboard, p_dst);
    }
};
