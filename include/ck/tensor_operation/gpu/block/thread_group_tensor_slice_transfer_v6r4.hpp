// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v6r4.hpp"

namespace ck {

template <typename ThreadGroup,
          typename ElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename SliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename Src0Data,
          typename Src1Data,
          typename Src2Data,
          typename DstData,
          typename Src0Desc,
          typename Src1Desc,
          typename Src2Desc,
          typename DstDesc,
          typename DimAccessOrder,
          index_t VectorDim,
          index_t ScalarPerVector,
          bool ThreadTransferSrc0ResetCoordinateAfterRun,
          bool ThreadTransferDstResetCoordinateAfterRun>
struct ThreadGroupTensorSliceTransfer_v6r4
{
    static constexpr index_t nDim = remove_reference_t<Src0Desc>::GetNumOfDimension();

    static constexpr auto thread_slice_lengths = SliceLengths{} / ThreadClusterLengths{};

    using Index = MultiIndex<nDim>;

    __device__ constexpr ThreadGroupTensorSliceTransfer_v6r4(const Src0Desc& src0_desc,
                                                             const Index& src0_block_slice_origin,
                                                             const DstDesc& dst_desc,
                                                             const Index& dst_block_slice_origin,
                                                             const ElementwiseOperation& element_op)
        : threadwise_transfer_(src0_desc,
                               make_zero_multi_index<nDim>(),
                               dst_desc,
                               make_zero_multi_index<nDim>(),
                               element_op)

    {
        static_assert(nDim == remove_cvref_t<Src0Desc>::GetNumOfDimension() &&
                          nDim == remove_cvref_t<DstDesc>::GetNumOfDimension() &&
                          nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == DimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<SliceLengths, decltype(thread_slice_lengths * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(ThreadGroup::GetNumOfThread() >= thread_cluster_desc_.GetElementSize(),
                      "wrong! ThreadGroup::GetNumOfThread() too small");

        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
                make_multi_index(get_thread_local_1d_id()));

            const auto thread_data_idx_begin = thread_cluster_idx * thread_slice_lengths;

            threadwise_transfer_.SetSrc0SliceOrigin(
                src0_desc, src0_block_slice_origin + thread_data_idx_begin);
            threadwise_transfer_.SetDstSliceOrigin(dst_desc,
                                                   dst_block_slice_origin + thread_data_idx_begin);
        }
    }

    template <typename Src0Buffer, typename Src1Buffer, typename Src2Buffer, 
            typename Src1SliceOriginIdx, typename Src2SliceOriginIdx, typename DstBuffer>
    __device__ void Run(const Src0Desc& src0_desc,
                        const Src0Buffer& src0_buf,
                        const Src1Desc& src1_desc,
                        const Src1SliceOriginIdx&,
                        const Src1Buffer& src1_buf,
                        const Src2Desc& src2_desc,
                        const Src2SliceOriginIdx&,
                        const Src2Buffer& src2_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.template Run<Src0Buffer, Src1Buffer, Src2Buffer,
             Src1SliceOriginIdx, Src2SliceOriginIdx, DstBuffer>(
                src0_desc, src0_buf, 
                src1_desc, Src1SliceOriginIdx{}, src1_buf, 
                src2_desc, Src2SliceOriginIdx{}, src2_buf, 
                dst_desc, dst_buf);
        }
    }

    __device__ void MoveSrc0SliceWindow(const Src0Desc& src0_desc, const Index& step)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrc0SliceWindow(src0_desc, step);
        }
    }

    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc, const Index& step)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveDstSliceWindow(dst_desc, step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    using ThreadwiseTransfer =
        ThreadwiseTensorSliceTransfer_v6r4<Src0Data,
                                           Src1Data,
                                           Src2Data,
                                           DstData,
                                           Src0Desc,
                                           Src1Desc,
                                           Src2Desc,
                                           DstDesc,
                                           ElementwiseOperation,
                                           decltype(thread_slice_lengths),
                                           DimAccessOrder,
                                           VectorDim,
                                           ScalarPerVector,
                                           DstInMemOp,
                                           ThreadTransferSrc0ResetCoordinateAfterRun,
                                           ThreadTransferDstResetCoordinateAfterRun>;

    ThreadwiseTransfer threadwise_transfer_;
};

} // namespace ck
