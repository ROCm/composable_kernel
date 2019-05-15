#pragma once
#include "threadwise_tensor_slice_op.hip.hpp"

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

    index_t mSrcMyThreadOffset;
    index_t mDstMyThreadOffset;

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

        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor(thread_cluster_lengths);

        // sanity check: data type
        static_assert(is_same<Float, float>::value, "wrong! only support float for now!\n");

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

        const auto thread_multi_id = thread_cluster_desc.GetMultiIndex(get_thread_local_1d_id());

        // compiler: thread_multi_id, src_data_multi_id, dst_data_multi_id, will use separate
        // regsiters, or only one copy???
        auto src_data_multi_id =
            reorder_array_given_old2new(thread_multi_id, map_thread_cluster_2_src_cluster);

        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr auto I    = decltype(IDim){};
            constexpr index_t i = I.Get();
            // compiler: will it really compute index here, or be merged with Get1dIndex and
            // optimized away???
            src_data_multi_id[i] *= src_sub_lengths.Get(I);
        });

        // compiler: will it really compute index here, or be merged with Get1dIndex and
        // optimized away???
        const auto dst_data_multi_id = reorder_array_given_new2old(src_data_multi_id, map_dst2src);

        mSrcMyThreadOffset = src_desc.Get1dIndex(src_data_multi_id + src_block_data_multi_id_begin);
        mDstMyThreadOffset = dst_desc.Get1dIndex(dst_data_multi_id + dst_block_data_multi_id_begin);
    }

    __device__ static constexpr index_t GetRegisterClipboardSize()
    {
        constexpr auto thread_sub_tensor_lengths = SrcSubLengths{};

        constexpr auto src_data_per_cluster_per_dims = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, SrcClusterLengths{});

        constexpr auto repeat_lengths =
            transform_sequences(mod_conv::integer_divide_ceiler<index_t>{},
                                SrcLengths{},
                                src_data_per_cluster_per_dims);

        constexpr auto thread_tensor_lengths = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, repeat_lengths);

        constexpr auto thread_tensor_desc = make_ConstantTensorDescriptor(thread_tensor_lengths);

        return thread_tensor_desc.GetElementSpace();
    }

    __device__ void RunLoadRegisterClipboard(const Float* __restrict__ p_src,
                                             Float* __restrict__ p_clipboard) const
    {
        constexpr auto thread_sub_tensor_lengths = SrcSubLengths{};

        constexpr auto src_data_per_cluster_per_dims = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, SrcClusterLengths{});

        constexpr auto repeat_lengths =
            transform_sequences(mod_conv::integer_divide_ceiler<index_t>{},
                                SrcLengths{},
                                src_data_per_cluster_per_dims);

        constexpr auto thread_tensor_lengths = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, repeat_lengths);

        constexpr auto thread_tensor_desc = make_ConstantTensorDescriptor(thread_tensor_lengths);

        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id_) {
            constexpr auto repeat_multi_id = decltype(repeat_multi_id_){};

            constexpr auto src_data_multi_id = transform_sequences(
                mod_conv::multiplies<index_t>{}, repeat_multi_id, src_data_per_cluster_per_dims);

            constexpr auto clipboard_data_multi_id = transform_sequences(
                mod_conv::multiplies<index_t>{}, repeat_multi_id, thread_sub_tensor_lengths);

            constexpr index_t src_offset = SrcDesc{}.Get1dIndex(src_data_multi_id);
            constexpr index_t clipboard_offset =
                thread_tensor_desc.Get1dIndex(clipboard_data_multi_id);

            threadwise_tensor_slice_copy(SrcDesc{},
                                         p_src + src_offset + mSrcMyThreadOffset,
                                         thread_tensor_desc,
                                         p_clipboard + clipboard_offset,
                                         thread_sub_tensor_lengths,
                                         Number<SrcDataPerRead>{});
        });
    }

    __device__ void RunStoreRegisterClipboard(const Float* __restrict__ p_clipboard,
                                              Float* __restrict__ p_dst) const
    {
        constexpr auto thread_sub_tensor_lengths = SrcSubLengths{};

        constexpr auto src_data_per_cluster_per_dims = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, SrcClusterLengths{});

        constexpr auto repeat_lengths =
            transform_sequences(mod_conv::integer_divide_ceiler<index_t>{},
                                SrcLengths{},
                                src_data_per_cluster_per_dims);

        constexpr auto thread_tensor_lengths = transform_sequences(
            mod_conv::multiplies<index_t>{}, thread_sub_tensor_lengths, repeat_lengths);

        constexpr auto thread_tensor_desc = make_ConstantTensorDescriptor(thread_tensor_lengths);

        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_multi_id_) {
            constexpr auto repeat_multi_id = decltype(repeat_multi_id_){};

            constexpr auto clipboard_data_multi_id = transform_sequences(
                mod_conv::multiplies<index_t>{}, repeat_multi_id, thread_sub_tensor_lengths);

            constexpr auto src_data_multi_id = transform_sequences(
                mod_conv::multiplies<index_t>{}, repeat_multi_id, src_data_per_cluster_per_dims);

            // reorder src_data_multi_id to get dst_data_multi_id
            constexpr auto dst_data_multi_id = src_data_multi_id.ReorderGivenNew2Old(MapDst2Src{});

            constexpr index_t clipboard_offset =
                thread_tensor_desc.Get1dIndex(clipboard_data_multi_id);

            constexpr index_t dst_offset = DstDesc{}.Get1dIndex(dst_data_multi_id);

// write in the order of dst
#if 1
            threadwise_tensor_slice_copy_reorder_given_dst2src_v2(thread_tensor_desc,
                                                                  p_clipboard + clipboard_offset,
                                                                  DstDesc{},
                                                                  p_dst + dst_offset +
                                                                      mDstMyThreadOffset,
                                                                  thread_sub_tensor_lengths,
                                                                  MapDst2Src{});
#else
            threadwise_tensor_slice_copy_reorder_given_dst2src_v3(thread_tensor_desc,
                                                                  p_clipboard + clipboard_offset,
                                                                  DstDesc{},
                                                                  p_dst + dst_offset +
                                                                      mDstMyThreadOffset,
                                                                  thread_sub_tensor_lengths,
                                                                  MapDst2Src{},
                                                                  Number<DstDataPerWrite>{});
#endif
        });
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        Float p_clipboard[GetRegisterClipboardSize()];

        RunLoadRegisterClipboard(p_src, p_clipboard);
        RunStoreRegisterClipboard(p_clipboard, p_dst);
    }
};
