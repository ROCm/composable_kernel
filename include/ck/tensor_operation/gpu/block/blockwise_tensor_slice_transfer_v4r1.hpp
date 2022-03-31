#ifndef CK_BLOCKWISE_TENSOR_SLICE_TRANSFER_V4R1_HPP
#define CK_BLOCKWISE_TENSOR_SLICE_TRANSFER_V4R1_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "cluster_descriptor.hpp"
#include "threadwise_tensor_slice_transfer_v3r1.hpp"

namespace ck {

// this version does following things to avoid scratch memory issue
// 1. Use StaticallyIndexedArray instead of C array for thread buffer
// 2. ThreadwiseTensorSliceTransfer_v3 does not keep reference to tensor descriptor
// 3. ThreadwiseTensorSliceTransfer_v3::Run() does not construct new tensor coordinate
template <index_t BlockSize,
          typename SrcElementwiseOperation,
          typename DstElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename BlockSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector,
          bool ThreadTransferSrcResetCoordinateAfterRun,
          bool ThreadTransferDstResetCoordinateAfterRun,
          index_t NumThreadScratch = 1>
struct BlockwiseTensorSliceTransfer_v4r1
{
    static constexpr index_t nDim = remove_reference_t<SrcDesc>::GetNumOfDimension();

    static constexpr auto thread_slice_lengths = BlockSliceLengths{} / ThreadClusterLengths{};

    using Index = MultiIndex<nDim>;

    __device__ constexpr BlockwiseTensorSliceTransfer_v4r1(
        const SrcDesc& src_desc,
        const Index& src_block_slice_origin,
        const SrcElementwiseOperation& src_element_op,
        const DstDesc& dst_desc,
        const Index& dst_block_slice_origin,
        const DstElementwiseOperation& dst_element_op)
        : threadwise_transfer_(src_desc,
                               make_zero_multi_index<nDim>(),
                               src_element_op,
                               dst_desc,
                               make_zero_multi_index<nDim>(),
                               dst_element_op)

    {
        static_assert(nDim == remove_reference_t<remove_cv_t<SrcDesc>>::GetNumOfDimension() &&
                          nDim == remove_reference_t<remove_cv_t<DstDesc>>::GetNumOfDimension() &&
                          nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(thread_slice_lengths * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(BlockSize >= thread_cluster_desc_.GetElementSize(),
                      "wrong! BlockSize too small");

        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
                make_multi_index(get_thread_local_1d_id()));

            const auto thread_data_idx_begin = thread_cluster_idx * thread_slice_lengths;

            threadwise_transfer_.SetSrcSliceOrigin(src_desc,
                                                   src_block_slice_origin + thread_data_idx_begin);
            threadwise_transfer_.SetDstSliceOrigin(dst_desc,
                                                   dst_block_slice_origin + thread_data_idx_begin);
        }
    }

    template <typename SrcBuffer, index_t ThreadScratchId = 0>
    __device__ void RunRead(const SrcDesc& src_desc,
                            const SrcBuffer& src_buf,
                            Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunRead(src_desc, src_buf, thread_scratch_id);
        }
    }

    template <typename DstBuffer, index_t ThreadScratchId = 0>
    __device__ void RunWrite(const DstDesc& dst_desc,
                             DstBuffer& dst_buf,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunWrite(dst_desc, dst_buf, thread_scratch_id);
        }
    }

    template <typename SrcBuffer, typename DstBuffer, index_t ThreadScratchId>
    __device__ void Run(const SrcDesc& src_desc,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf,
                        Number<ThreadScratchId> thread_scratch_id)
    {
        RunRead(src_desc, src_buf, thread_scratch_id);
        RunWrite(dst_desc, dst_buf, thread_scratch_id);
    }

    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrcSliceWindow(src_desc, step);
        }
    }

    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveDstSliceWindow(dst_desc, step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    using ThreadwiseTransfer =
        ThreadwiseTensorSliceTransfer_v3r1<decltype(thread_slice_lengths),
                                           SrcElementwiseOperation,
                                           DstElementwiseOperation,
                                           DstInMemOp,
                                           SrcData,
                                           DstData,
                                           SrcDesc,
                                           DstDesc,
                                           SrcDimAccessOrder,
                                           DstDimAccessOrder,
                                           SrcVectorDim,
                                           DstVectorDim,
                                           SrcScalarPerVector,
                                           DstScalarPerVector,
                                           SrcScalarStrideInVector,
                                           DstScalarStrideInVector,
                                           ThreadTransferSrcResetCoordinateAfterRun,
                                           ThreadTransferDstResetCoordinateAfterRun,
                                           NumThreadScratch>;

    ThreadwiseTransfer threadwise_transfer_;
};

} // namespace ck
#endif
