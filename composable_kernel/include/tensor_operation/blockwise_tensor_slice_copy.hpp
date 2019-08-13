#ifndef CK_BLOCKWISE_TENSOR_SLICE_COPY_HPP
#define CK_BLOCKWISE_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "threadwise_tensor_slice_copy.hpp"

namespace ck {

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
struct BlockwiseTensorSliceReorderCopy_v3
{
    static constexpr index_t nDim = SrcLengths::GetSize();

    index_t mThreadSrcOffset;
    index_t mThreadDstOffset;

    __device__
    BlockwiseTensorSliceReorderCopy_v3(Array<index_t, nDim> src_block_data_multi_id_begin,
                                       Array<index_t, nDim> dst_block_data_multi_id_begin)
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

        constexpr auto thread_cluster_desc =
            make_ConstantTensorDescriptor_packed(thread_cluster_lengths);

        // sanity check: data type
        static_assert(is_same<Float, float>{}, "wrong! only support float for now!\n");

        // sanity check: nDim
        static_assert(SrcDesc::GetNumOfDimension() == nDim &&
                          DstDesc::GetNumOfDimension() == nDim && SrcLengths::GetSize() == nDim &&
                          SrcSubLengths::GetSize() == nDim &&
                          SrcClusterLengths::GetSize() == nDim && MapDst2Src::GetSize() == nDim &&
                          MapThreadCluster2SrcCluster::GetSize() == nDim,
                      "wrong! nDim is not consistent\n");

        // sanity check: BlockSize
        constexpr index_t num_active_thread = thread_cluster_desc.GetElementSize();

        static_assert(BlockSize >= num_active_thread,
                      "wrong! BlockSize is not big enough for ThreadPerDims!");

        // sanity check: work division
        static_for<0, nDim, 1>{}([&](auto IDim) {
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

        const auto thread_multi_id =
            thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        // compiler: thread_multi_id, src_data_multi_id, dst_data_multi_id, will use separate
        // regsiters, or only one copy???
        auto src_data_multi_id =
            reorder_array_given_old2new(thread_multi_id, map_thread_cluster_2_src_cluster);

        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr index_t idim = IDim;
            // compiler: will it really compute index here, or be merged with
            // GetOffsetFromMultiIndex and
            // optimized away???
            src_data_multi_id(idim) *= src_sub_lengths.Get(IDim);
        });

        // compiler: will it really compute index here, or be merged with GetOffsetFromMultiIndex
        // and
        // optimized away???
        const auto dst_data_multi_id = reorder_array_given_new2old(src_data_multi_id, map_dst2src);

        mThreadSrcOffset =
            src_desc.GetOffsetFromMultiIndex(src_data_multi_id + src_block_data_multi_id_begin);

        mThreadDstOffset =
            dst_desc.GetOffsetFromMultiIndex(dst_data_multi_id + dst_block_data_multi_id_begin);
#if 0
        if(get_block_1d_id() == 0 && get_thread_local_1d_id() == 0)
        {
            print_ConstantTensorDescriptor(thread_cluster_desc, "thread_cluster_desc: ");
        }

        if(get_block_1d_id() == 0)
        {
            printf("id %5u %5u: "
                   "thread_multi_id: %u %u, "
                   "src_block_data_multi_id_begin: %u %u, "
                   "src_data_multi_id: %u %u, "
                   "mThreadSrcOffset %u, mThreadDstOffset %u \n",
                   get_block_1d_id(),
                   get_thread_local_1d_id(),
                   thread_multi_id[0],
                   thread_multi_id[1],
                   src_block_data_multi_id_begin[0],
                   src_block_data_multi_id_begin[1],
                   src_data_multi_id[0],
                   src_data_multi_id[1],
                   mThreadSrcOffset,
                   mThreadDstOffset);
        }
#endif
    }

    __device__ static constexpr index_t GetRegisterBufferSize()
    {
        constexpr auto thread_sub_tensor_lengths = SrcSubLengths{};

        constexpr auto src_data_per_cluster_per_dims =
            thread_sub_tensor_lengths * SrcClusterLengths{};

        constexpr auto repeat_lengths = transform_sequences(
            math::integer_divide_ceiler<index_t>{}, SrcLengths{}, src_data_per_cluster_per_dims);

        constexpr auto thread_tensor_lengths = thread_sub_tensor_lengths * repeat_lengths;

        constexpr auto thread_tensor_desc =
            make_ConstantTensorDescriptor_packed(thread_tensor_lengths);

        return thread_tensor_desc.GetElementSpace();
    }

    __device__ void RunLoadRegisterBuffer(const Float* __restrict__ p_src,
                                          Float* __restrict__ p_clipboard) const
    {
        constexpr auto thread_sub_tensor_lengths = SrcSubLengths{};

        constexpr auto src_data_per_cluster_per_dims =
            thread_sub_tensor_lengths * SrcClusterLengths{};

        constexpr auto repeat_lengths = transform_sequences(
            math::integer_divide_ceiler<index_t>{}, SrcLengths{}, src_data_per_cluster_per_dims);

        constexpr auto thread_tensor_lengths = thread_sub_tensor_lengths * repeat_lengths;

        constexpr auto thread_tensor_desc =
            make_ConstantTensorDescriptor_packed(thread_tensor_lengths);

        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id_) {
            constexpr auto repeat_multi_id = decltype(repeat_multi_id_){};

            constexpr auto src_data_multi_id = repeat_multi_id * src_data_per_cluster_per_dims;

            constexpr auto clipboard_data_multi_id = repeat_multi_id * thread_sub_tensor_lengths;

            constexpr index_t src_offset = SrcDesc{}.GetOffsetFromMultiIndex(src_data_multi_id);
            constexpr index_t clipboard_offset =
                thread_tensor_desc.GetOffsetFromMultiIndex(clipboard_data_multi_id);

            threadwise_tensor_slice_copy(SrcDesc{},
                                         p_src + src_offset + mThreadSrcOffset,
                                         thread_tensor_desc,
                                         p_clipboard + clipboard_offset,
                                         thread_sub_tensor_lengths,
                                         Number<SrcDataPerRead>{});
        });
    }

    __device__ void RunStoreRegisterBuffer(const Float* __restrict__ p_clipboard,
                                           Float* __restrict__ p_dst) const
    {
        constexpr auto thread_sub_tensor_lengths = SrcSubLengths{};

        constexpr auto src_data_per_cluster_per_dims =
            thread_sub_tensor_lengths * SrcClusterLengths{};

        constexpr auto repeat_lengths = transform_sequences(
            math::integer_divide_ceiler<index_t>{}, SrcLengths{}, src_data_per_cluster_per_dims);

        constexpr auto thread_tensor_lengths = thread_sub_tensor_lengths * repeat_lengths;

        constexpr auto thread_tensor_desc =
            make_ConstantTensorDescriptor_packed(thread_tensor_lengths);

        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id_) {
            constexpr auto repeat_multi_id = decltype(repeat_multi_id_){};

            constexpr auto clipboard_data_multi_id = repeat_multi_id * thread_sub_tensor_lengths;

            constexpr auto src_data_multi_id = repeat_multi_id * src_data_per_cluster_per_dims;

            // reorder src_data_multi_id to get dst_data_multi_id
            constexpr auto dst_data_multi_id = src_data_multi_id.ReorderGivenNew2Old(MapDst2Src{});

            constexpr index_t clipboard_offset =
                thread_tensor_desc.GetOffsetFromMultiIndex(clipboard_data_multi_id);

            constexpr index_t dst_offset = DstDesc{}.GetOffsetFromMultiIndex(dst_data_multi_id);

// write in the order of dst
#if 1
            threadwise_tensor_slice_copy_reorder_given_dst2src_v2(thread_tensor_desc,
                                                                  p_clipboard + clipboard_offset,
                                                                  DstDesc{},
                                                                  p_dst + dst_offset +
                                                                      mThreadDstOffset,
                                                                  thread_sub_tensor_lengths,
                                                                  MapDst2Src{});
#else
            threadwise_tensor_slice_copy_reorder_given_dst2src_v3(thread_tensor_desc,
                                                                  p_clipboard + clipboard_offset,
                                                                  DstDesc{},
                                                                  p_dst + dst_offset +
                                                                      mThreadDstOffset,
                                                                  thread_sub_tensor_lengths,
                                                                  MapDst2Src{},
                                                                  Number<DstDataPerWrite>{});
#endif
        });
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        Float p_clipboard[GetRegisterBufferSize()];

        RunLoadRegisterBuffer(p_src, p_clipboard);
        RunStoreRegisterBuffer(p_clipboard, p_dst);
    }

    // this function doesn't do santiy check on whether the slicing window is out of the boundary
    // of the tensor being sliced
    template <index_t IDim_, index_t StepSize, bool PositiveDirection>
    __device__ void MoveSlicingWindowOnSourceTensor(
        Number<IDim_>, Number<StepSize>, integral_constant<bool, PositiveDirection> direction)
    {
        constexpr auto IDim = Number<IDim_>{};

        static_if<PositiveDirection>{}([&](auto fwd) {
            mThreadSrcOffset += StepSize * fwd(SrcDesc{}).GetStride(IDim);
        }).Else([&](auto fwd) { mThreadSrcOffset -= StepSize * fwd(SrcDesc{}).GetStride(IDim); });
    }
};

} // namespace ck
#endif
