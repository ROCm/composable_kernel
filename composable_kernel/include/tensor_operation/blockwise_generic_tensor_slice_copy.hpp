#ifndef CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"
#include "tensor_coordinate.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"

#ifndef CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1 1
#endif

namespace ck {

// slice a (normal or merged) tensor, and copy it into another (normal or merged) tensor
// memory layout (ordering of dimensions) can be different between src and dst.
// on a merged dimension that constains multiple original dimensions,
// its sub-length need to evenly divide the length of the last original dimension
// so each thread is effectively reading a normal (not merged) tensor
template <index_t BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class SliceLengths,
          class SubLengths,
          class ThreadClusterLengths,
          class ThreadClusterArrangeOrder,
          class SrcAccessOrder,
          class DstAccessOrder,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite>
struct BlockwiseGenericTensorSliceCopy_v1
{
    static constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    static constexpr index_t nOriginalDimSrc =
        SrcDesc::GetOriginalTensorDescriptor().GetNumOfDimension();
    static constexpr index_t nOriginalDimDst =
        DstDesc::GetOriginalTensorDescriptor().GetNumOfDimension();

    // per-thread offset
    index_t mThreadSrcOffset;
    index_t mThreadDstOffset;

    // "mThreadSrcOriginalMultiId", "mThreadSrcPartialOffsets, "mThreadDstOriginalMultiId",
    // "mThreadDstPartialOffsets" are always calculated inside constructor, and would be
    // updated if slicing-window is moved. However, they will not be used if you always move
    // the slicing-window along a non-merged dimension. In that case, compiler should be
    // able to remove these calculation.
    // TODO: make sure compiler would actually remove them in that case

    // partial offset in each (merged) dimension
    Array<index_t, nDim> mThreadSrcPartialOffsets;
    Array<index_t, nDim> mThreadDstPartialOffsets;

    // multi-id of original tensor
    Array<index_t, nOriginalDimSrc> mThreadSrcOriginalMultiId;
    Array<index_t, nOriginalDimDst> mThreadDstOriginalMultiId;

    __device__
    BlockwiseGenericTensorSliceCopy_v1(Array<index_t, nDim> src_block_data_multi_id_begin,
                                       Array<index_t, nDim> dst_block_data_multi_id_begin)
    {
        // check NDim consistency
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::GetSize() &&
                          nDim == SubLengths::GetSize() &&
                          nDim == ThreadClusterLengths::GetSize() &&
                          nDim == ThreadClusterArrangeOrder::GetSize() &&
                          nDim == SrcAccessOrder::GetSize() && nDim == DstAccessOrder::GetSize(),
                      "wrong");

        // check thread arrange order and read/write access order are valid
        static_assert(is_valid_sequence_map<ThreadClusterArrangeOrder>::value &&
                          is_valid_sequence_map<SrcAccessOrder>::value &&
                          is_valid_sequence_map<DstAccessOrder>::value,
                      "wrong!");

        // thread cluster
        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor_packed(
            ThreadClusterLengths::ReorderGivenNew2Old(ThreadClusterArrangeOrder{}));

        // BlockSize
        static_assert(BlockSize == thread_cluster_desc.GetElementSize(), "wrong! BlockSize");

        // divide work
        constexpr auto data_per_cluster_per_dims = SubLengths{} * ThreadClusterLengths{};

        static_for<0, nDim, 1>{}([&](auto IDim) {
            static_assert(SliceLengths::Get(IDim) % SubLengths::Get(IDim) == 0,
                          "wrong! cannot evenly divide sliced tensor into sub-tensor");

            static_assert(SliceLengths::Get(IDim) % data_per_cluster_per_dims.Get(IDim) == 0,
                          "wrong! cannot evenly divide sliced tensor into cluster");
        });

        // on a merged dimension that constains multiple original dimensions,
        // its sub-length need to evenly divide the length of the last original dimension,
        // so each thread is effectively reading a normal (not merged) tensor
        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr auto sub_length = SubLengths::Get(IDim);

            constexpr auto idim_original_src = SrcDesc::GetContainedOriginalDimensions(IDim).Back();
            static_assert(SrcDesc::GetOriginalTensorDescriptor().GetLength(idim_original_src) %
                                  sub_length ==
                              0,
                          "wrong!");

            constexpr auto idim_original_dst = DstDesc::GetContainedOriginalDimensions(IDim).Back();
            static_assert(DstDesc::GetOriginalTensorDescriptor().GetLength(idim_original_dst) %
                                  sub_length ==
                              0,
                          "wrong!");
        });

        // calculate mThreadSrcOffset, mThreadDstOffset
        const auto thread_cluster_multi_id =
            thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        const auto data_cluster_multi_id =
            reorder_array_given_old2new(thread_cluster_multi_id, ThreadClusterArrangeOrder{});

        const auto thread_data_multi_id_begin = data_cluster_multi_id * SubLengths{};

        // original multi-id
        mThreadSrcOriginalMultiId = SrcDesc::GetOriginalMultiIndexFromMultiIndex(
            src_block_data_multi_id_begin + thread_data_multi_id_begin);

        mThreadDstOriginalMultiId = DstDesc::GetOriginalMultiIndexFromMultiIndex(
            dst_block_data_multi_id_begin + thread_data_multi_id_begin);

        // partial offset on each dimension
        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr auto src_partial_original_dims =
                SrcDesc::GetContainedOriginalDimensions(IDim);

            constexpr auto src_partial_original_desc =
                SrcDesc::GetOriginalTensorDescriptor().Extract(src_partial_original_dims);

            mThreadSrcPartialOffsets(IDim) = src_partial_original_desc.GetOffsetFromMultiIndex(
                extract_array(mThreadSrcOriginalMultiId, src_partial_original_dims));
        });

        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr auto dst_partial_original_dims =
                DstDesc::GetContainedOriginalDimensions(IDim);

            constexpr auto dst_partial_original_desc =
                DstDesc::GetOriginalTensorDescriptor().Extract(dst_partial_original_dims);

            mThreadDstPartialOffsets(IDim) = dst_partial_original_desc.GetOffsetFromMultiIndex(
                extract_array(mThreadDstOriginalMultiId, dst_partial_original_dims));
        });

        // complete offset
        mThreadSrcOffset = accumulate_on_array(
            mThreadSrcPartialOffsets, math::plus<index_t>{}, static_cast<index_t>(0));

        mThreadDstOffset = accumulate_on_array(
            mThreadDstPartialOffsets, math::plus<index_t>{}, static_cast<index_t>(0));
    }

    __device__ static constexpr index_t GetRegisterBufferSize()
    {
        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * ThreadClusterLengths{});

        constexpr auto thread_tensor_desc =
            make_ConstantTensorDescriptor_packed(SubLengths{} * repeat_lengths);

        return thread_tensor_desc.GetElementSpace();
    }

    __device__ void RunLoadRegisterBuffer(const Float* __restrict__ p_src,
                                          Float* __restrict__ p_Buffer) const
    {
        constexpr auto thread_sub_tensor_lengths = SubLengths{};

        constexpr auto data_per_cluster_per_dims =
            thread_sub_tensor_lengths * ThreadClusterLengths{};

        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * ThreadClusterLengths{});

        constexpr auto thread_tensor_desc =
            make_ConstantTensorDescriptor_packed(thread_sub_tensor_lengths * repeat_lengths);

#if CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1
        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id) {
            constexpr auto src_thread_data_multi_id_begin =
                repeat_multi_id * data_per_cluster_per_dims;

            constexpr auto Buffer_data_multi_id_begin = repeat_multi_id * thread_sub_tensor_lengths;

            constexpr index_t src_offset =
                SrcDesc::GetOffsetFromMultiIndex(src_thread_data_multi_id_begin);

            constexpr index_t Buffer_offset =
                thread_tensor_desc.GetOffsetFromMultiIndex(Buffer_data_multi_id_begin);
#else
        ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id) {
            const auto src_thread_data_multi_id_begin = repeat_multi_id * data_per_cluster_per_dims;

            const auto Buffer_data_multi_id_begin = repeat_multi_id * thread_sub_tensor_lengths;

            const index_t src_offset =
                SrcDesc::GetOffsetFromMultiIndex(src_thread_data_multi_id_begin);

            const index_t Buffer_offset =
                thread_tensor_desc.GetOffsetFromMultiIndex(Buffer_data_multi_id_begin);
#endif

            // By position the origin of the per-thread window at the point, where multi-index
            // of the SrcDesc (might be a merged tensor) is all-zero. This threadwise slice copy
            // is assuming each thread is copy a noraml (not merged) tensor.
            // User need to guarantee this is true.
            // By setting SubLengths = 1 at the merged dimension, this is always true;
            // If in the future, you want to enable SubLengths > 1 at the merged dimension,
            // special care in implementation is needed
            threadwise_generic_tensor_slice_copy_v1(SrcDesc{},
                                                    p_src + src_offset + mThreadSrcOffset,
                                                    make_zero_array<index_t, nDim>(),
                                                    thread_tensor_desc,
                                                    p_Buffer + Buffer_offset,
                                                    make_zero_array<index_t, nDim>(),
                                                    thread_sub_tensor_lengths,
                                                    SrcAccessOrder{},
                                                    Number<SrcDataPerRead>{});
        });
    }

    __device__ void RunStoreRegisterBuffer(const Float* __restrict__ p_Buffer,
                                           Float* __restrict__ p_dst) const
    {
        constexpr auto thread_sub_tensor_lengths = SubLengths{};

        constexpr auto data_per_cluster_per_dims =
            thread_sub_tensor_lengths * ThreadClusterLengths{};

        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * ThreadClusterLengths{});

        constexpr auto thread_tensor_desc =
            make_ConstantTensorDescriptor_packed(thread_sub_tensor_lengths * repeat_lengths);

#if CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1
        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id) {
            constexpr auto Buffer_data_multi_id_begin = repeat_multi_id * thread_sub_tensor_lengths;

            constexpr auto dst_data_multi_id_begin = repeat_multi_id * data_per_cluster_per_dims;

            constexpr index_t Buffer_offset =
                thread_tensor_desc.GetOffsetFromMultiIndex(Buffer_data_multi_id_begin);

            constexpr index_t dst_offset =
                DstDesc::GetOffsetFromMultiIndex(dst_data_multi_id_begin);
#else
        ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id) {
            const auto Buffer_data_multi_id_begin = repeat_multi_id * thread_sub_tensor_lengths;

            const auto dst_data_multi_id_begin = repeat_multi_id * data_per_cluster_per_dims;

            const index_t Buffer_offset =
                thread_tensor_desc.GetOffsetFromMultiIndex(Buffer_data_multi_id_begin);

            const index_t dst_offset = DstDesc::GetOffsetFromMultiIndex(dst_data_multi_id_begin);
#endif

            // By position the origin of the per-thread window at the point, where multi-index
            // of the SrcDesc (might be a merged tensor) is all-zero. This threadwise slice copy
            // is assuming each thread is copy a noraml (not merged) tensor.
            // User need to guarantee this is true.
            // By setting SubLengths = 1 at the merged dimension, this is always true;
            // If in the future, you want to enable SubLengths > 1 at the merged dimension,
            // special care in implementation is needed
            threadwise_generic_tensor_slice_copy_v1(thread_tensor_desc,
                                                    p_Buffer + Buffer_offset,
                                                    make_zero_array<index_t, nDim>(),
                                                    DstDesc{},
                                                    p_dst + dst_offset + mThreadDstOffset,
                                                    make_zero_array<index_t, nDim>(),
                                                    thread_sub_tensor_lengths,
                                                    DstAccessOrder{},
                                                    Number<DstDataPerWrite>{});
        });
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        Float p_Buffer[GetRegisterBufferSize()];

        RunLoadRegisterBuffer(p_src, p_Buffer);
        RunStoreRegisterBuffer(p_Buffer, p_dst);
    }

    // When moving the slicing windows along a merged dimension, if the strides of the
    // contained (by the merged dimension) original dimensions are not in descending order,
    // then there is no guarantee that the new offset will be larger than the old offset
    // for movement in positive direction (vice versue for movement in negative direction).
    // As a result, there is the possiblity that the offset calculation may result in
    // unsigned integer underflow (due to "-" operation). However, this hazard should not
    // happen, as long as the users make sure the slicing window would not be moved out of
    // the boundary of the tensor being sliced. This functions doesn't do runtime sanity
    // check on out-of-bound slicing window, for performance reason
    template <index_t IDim_, index_t StepSize, bool PositiveDirection>
    __device__ void MoveSlicingWindowOnSourceTensor(
        Number<IDim_>, Number<StepSize>, integral_constant<bool, PositiveDirection> direction)
    {
        constexpr auto IDim = Number<IDim_>{};

        static_if<SrcDesc::ContainMultipleOriginalDimensions(IDim)>{}([&](auto) {
            // logic for a merged dimension, also works for non-merged dimension, but its logic may
            // be unncessarily complicated for compiler to remove calculations that are useless for
            // a non-merged dimension

            // extract partial original dimensions
            constexpr auto src_partial_original_dims =
                SrcDesc::GetContainedOriginalDimensions(IDim);

            constexpr auto src_partial_original_desc =
                SrcDesc::GetOriginalTensorDescriptor().Extract(src_partial_original_dims);

            // calculate new partial original multi-id
            auto old_src_partial_original_multi_id =
                extract_array(mThreadSrcOriginalMultiId, src_partial_original_dims);

            auto new_src_partial_original_multi_id =
                src_partial_original_desc.UpdateMultiIndexGivenStepSizeOf1dIndex(
                    old_src_partial_original_multi_id, StepSize, direction);

            // update "mThreadSrcOriginalMultiId"
            static_for<0, decltype(src_partial_original_dims)::GetSize(), 1>{}([&](auto I) {
                constexpr auto IDimOriginal = src_partial_original_dims[I];

                mThreadSrcOriginalMultiId(IDimOriginal) = new_src_partial_original_multi_id[I];
            });

            // calculate new partial offset on this merged dimension
            const index_t old_src_partial_offset = mThreadSrcPartialOffsets[IDim];

            const index_t new_src_partial_offset =
                src_partial_original_desc.GetOffsetFromMultiIndex(
                    new_src_partial_original_multi_id);

            // update "mThreadSrcPartialOffsets"
            mThreadSrcPartialOffsets(IDim) = new_src_partial_offset;

            // update "mThreadSrcOffset", do "+" before "-" to avoid underflow
            mThreadSrcOffset = (mThreadSrcOffset + new_src_partial_offset) - old_src_partial_offset;
        }).Else([&](auto) {
            // Logic for non-merged dimension. If you are never going to move the slicing window on
            // a merged dimension, then "mThreadSrcOriginalMultiId" and "mThreadSrcPartialOffsets",
            // which are being calculated here, will never be used later. In this case, compiler
            // should be able to remove these calculations.
            // TODO: make sure compiler would actually remove them in this case.

            // It is the user's responsiblity to make sure the slicing window will not be moved out
            // of the boundary of the tensor being sliced. Otherwise, there might be hazard like
            // unsigned integer underflow. That is NO runtime sanity check to prevent the hazard

            constexpr auto IDimOriginal = SrcDesc::GetContainedOriginalDimensions(IDim).Front();

            static_if<PositiveDirection>{}([&](auto fwd) {
                mThreadSrcOffset += StepSize * fwd(SrcDesc{}).GetStride(IDim);

                mThreadSrcOriginalMultiId(IDimOriginal) += StepSize;

                mThreadSrcPartialOffsets(IDim) += StepSize * fwd(SrcDesc{}).GetStride(IDim);
            }).Else([&](auto fwd) {
                mThreadSrcOffset -= StepSize * fwd(SrcDesc{}).GetStride(IDim);

                mThreadSrcOriginalMultiId(IDimOriginal) -= StepSize;

                mThreadSrcPartialOffsets(IDim) -= StepSize * fwd(SrcDesc{}).GetStride(IDim);
            });
        });
    }
};

template <index_t BlockSize,
          class TData,
          class SrcDesc,
          class DstDesc,
          class SrcCoordinate,
          class DstCoordinate,
          class SliceLengths,
          class SubLengths,
          class ThreadClusterLengths,
          class ThreadClusterArrangeOrder>
struct BlockwiseGenericTensorSliceCopy_v2
{
    static constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    __device__ constexpr BlockwiseGenericTensorSliceCopy_v2(SrcCoordinate src_block_slice_origin,
                                                            DstCoordinate dst_block_slice_origin)
    {
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::GetSize() &&
                          nDim == SubLengths::GetSize() &&
                          nDim == ThreadClusterLengths::GetSize() &&
                          nDim == ThreadClusterArrangeOrder::GetSize(),
                      "wrong! nDim not consistent");

        static_assert(is_same<SliceLengths, decltype(SubLengths{} * ThreadClusterLengths{})>{},
                      "wrong! threads should be mapped to cover entire slicing window");

        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor_packed(
            ThreadClusterLengths::ReorderGivenNew2Old(ThreadClusterArrangeOrder{}));

        static_assert(BlockSize == thread_cluster_desc.GetElementSize(),
                      "wrong! BlockSize not consistent with ThreadClusterLengths");

        const auto thread_cluster_multi_id =
            thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        const auto data_cluster_multi_id =
            reorder_array_given_old2new(thread_cluster_multi_id, ThreadClusterArrangeOrder{});

        const auto thread_data_multi_id_begin = data_cluster_multi_id * SubLengths{};

        mThreadwiseLoad.SetSrcSliceOrigin(src_block_slice_origin + thread_data_multi_id_begin);
        mThreadwiseLoad.SetDstSliceOrigin(make_zero_array<index_t, nDim>());

        mThreadwiseStore.SetSrcSliceOrigin(make_zero_array<index_t, nDim>());
        mThreadwiseStore.SetDstSliceOrigin(dst_block_slice_origin + thread_data_multi_id_begin);
    }

    __device__ static constexpr index_t GetRegisterBufferSize()
    {
        return RegisterBufferDesc::GetElementSpace();
    }

    __device__ void RunLoadRegisterBuffer(const TData* p_src, TData* p_buffer) const
    {
        mThreadwiseLoad.Run(p_src, p_buffer);
    }

    __device__ void RunStoreRegisterBuffer(const TData* p_buffer, TData* p_dst) const
    {
        mThreadwiseStore.Run(p_buffer, p_dst);
    }

    __device__ void Run(const TData* p_src, TData* p_dst) const
    {
        TData p_buffer[GetRegisterBufferSize()];

        mThreadwiseLoad.Run(p_src, p_buffer);
        mThreadwiseStore.Run(p_buffer, p_dst);
    }

    __device__ void MoveSrcSlicingWindow(Array<index_t, nDim> step_sizes, bool positive_direction)
    {
        mThreadwiseLoad.MoveSrcSlicingWindow(step_sizes, positive_direction);
    }

    __device__ void MoveDstSlicingWindow(Array<index_t, nDim> step_sizes, bool positive_direction)
    {
        mThreadwiseStore.MoveDstSlicingWindow(step_sizes, positive_direction);
    }

    private:
    using RegisterBufferDesc = decltype(make_ConstantTensorDescriptor_packed(SubLengths{}));

    using ThreadwiseLoad =
        ThreadwiseGenericTensorSliceCopy_v2<TData,
                                            SrcDesc,
                                            RegisterBufferDesc,
                                            SrcCoordinate,
                                            NormalTensorCoordinate<RegisterBufferDesc>,
                                            SubLengths>;

    using ThreadwiseStore =
        ThreadwiseGenericTensorSliceCopy_v2<TData,
                                            RegisterBufferDesc,
                                            DstDesc,
                                            NormalTensorCoordinate<RegisterBufferDesc>,
                                            DstCoordinate,
                                            SubLengths>;
    ThreadwiseLoad mThreadwiseLoad;
    ThreadwiseStore mThreadwiseStore;
};

} // namespace ck

#endif
