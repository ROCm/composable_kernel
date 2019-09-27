#ifndef CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_coordinate.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"

namespace ck {

template <index_t BlockSize,
          typename BlockSrcDesc,
          typename BlockDstDesc,
          typename BlockSliceLengths,
          typename ThreadSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorAccessDim,
          index_t DstVectorAccessDim,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
struct BlockwiseGenericTensorSliceCopy_v4
{
    static constexpr index_t nDim = BlockSrcDesc::GetNumOfDimension();
    using Index                   = MultiIndex<nDim>;

    __device__ constexpr BlockwiseGenericTensorSliceCopy_v4(const Index& src_block_slice_origin,
                                                            const Index& dst_block_slice_origin)
    {
        static_assert(nDim == BlockSrcDesc::GetNumOfDimension() &&
                          nDim == BlockDstDesc::GetNumOfDimension() &&
                          nDim == BlockSliceLengths::Size() && nDim == ThreadSliceLengths::Size() &&
                          nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(ThreadSliceLengths{} * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        // map threads to cluster
        constexpr auto thread_cluster_desc =
            make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

        static_assert(BlockSize == thread_cluster_desc.GetElementSize(),
                      "wrong! BlockSize not consistent with ThreadClusterLengths");

        const auto thread_cluster_id =
            thread_cluster_desc.CalculateClusterIndex(get_thread_local_1d_id());

        const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};

        mThreadwiseLoad.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
        mThreadwiseLoad.SetDstSliceOrigin(make_zero_array<index_t, nDim>());

        mThreadwiseStore.SetSrcSliceOrigin(make_zero_array<index_t, nDim>());
        mThreadwiseStore.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
    }

    __device__ static constexpr index_t GetThreadBufferSize()
    {
        return ThreadBufferDesc::GetElementSpace();
    }

    template <typename BlockSrcData,
              typename ThreadBufferData,
              address_space_t BlockSrcAddressSpace     = address_space_t::generic,
              address_space_t ThreadBufferAddressSpace = address_space_t::generic>
    __device__ void RunLoadThreadBuffer(const BlockSrcData* p_block_src,
                                        ThreadBufferData* p_thread_buffer) const
    {
#if 0
        mThreadwiseLoad.template Run<BlockSrcData,
                                             ThreadBufferData,
                                             BlockSrcAddressSpace,
                                             ThreadBufferAddressSpace>(p_block_src,
                                                                       p_thread_buffer);
#else // tweaking
        mThreadwiseLoad.template Run_optimized_src_address_calculation<BlockSrcData,
                                                                       ThreadBufferData,
                                                                       BlockSrcAddressSpace,
                                                                       ThreadBufferAddressSpace>(
            p_block_src, p_thread_buffer);
#endif
    }

    template <typename ThreadBufferData,
              typename BlockDstData,
              address_space_t ThreadBufferAddressSpace = address_space_t::generic,
              address_space_t BlockDstAddressSpace     = address_space_t::generic>
    __device__ void RunStoreThreadBuffer(const ThreadBufferData* p_thread_buffer,
                                         BlockDstData* p_block_dst) const
    {
#if 0
        mThreadwiseStore.template Run<ThreadBufferData,
                                              BlockDstData,
                                              ThreadBufferAddressSpace,
                                              BlockDstAddressSpace>(p_thread_buffer, p_block_dst);
#else // tweaking
        mThreadwiseStore.template Run_optimized_dst_address_calculation<ThreadBufferData,
                                                                        BlockDstData,
                                                                        ThreadBufferAddressSpace,
                                                                        BlockDstAddressSpace>(
            p_thread_buffer, p_block_dst);
#endif
    }

    template <typename BlockSrcData,
              typename BlockDstData,
              address_space_t BlockSrcAddressSpace = address_space_t::generic,
              address_space_t BlockDstAddressSpace = address_space_t::generic>
    __device__ void Run(const BlockSrcData* p_block_src, BlockDstData* p_block_dst) const
    {
        BlockSrcData p_thread_buffer[GetThreadBufferSize()];

        RunLoadThreadBuffer<BlockSrcData,
                            BlockSrcData,
                            BlockSrcAddressSpace,
                            address_space_t::generic>(p_block_src, p_thread_buffer);
        RunStoreThreadBuffer<BlockSrcData,
                             BlockDstData,
                             address_space_t::generic,
                             BlockDstAddressSpace>(p_thread_buffer, p_block_dst);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveSrcSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseLoad.MoveSrcSliceWindow(step_sizes, positive_direction);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveDstSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseStore.MoveDstSliceWindow(step_sizes, positive_direction);
    }

    private:
    using ThreadBufferDesc = decltype(make_native_tensor_descriptor_packed(ThreadSliceLengths{}));

    using ThreadwiseLoad = ThreadwiseGenericTensorSliceCopy_v4r2<BlockSrcDesc,
                                                                 ThreadBufferDesc,
                                                                 ThreadSliceLengths,
                                                                 SrcDimAccessOrder,
                                                                 SrcVectorAccessDim,
                                                                 SrcDataPerAccess,
                                                                 1>;

    using ThreadwiseStore = ThreadwiseGenericTensorSliceCopy_v4r2<ThreadBufferDesc,
                                                                  BlockDstDesc,
                                                                  ThreadSliceLengths,
                                                                  DstDimAccessOrder,
                                                                  DstVectorAccessDim,
                                                                  1,
                                                                  DstDataPerAccess>;

    ThreadwiseLoad mThreadwiseLoad;
    ThreadwiseStore mThreadwiseStore;
};

} // namespace ck

#endif
