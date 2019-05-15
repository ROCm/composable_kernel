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
          class ThreadArrangeOrder,
          class SrcAccessOrder,
          class DstAccessOrder>
struct BlockwiseTensorSliceCopy_generic_v1
{
    static constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    index_t mSrcMyThreadOffset;
    index_t mDstMyThreadOffset;

    __device__ BlockwiseTensorSliceCopy_generic_v1(Array<index_t, nDim> src_block_multi_id_offset,
                                                   Array<index_t, nDim> dst_block_multi_id_offset)
    {
        // only support SrcSubLengths.GetLength() == 1 on merged dimension, for now
        // check SrcDataPerRead should be 1, if last dimension is a merged dimension

        // check NDim consistent

        // calculate mSrcMyThreadOffset
        // calculate mDstMyThreadOffset
    }

    __device__ static constexpr index_t GetRegisterClipboardSize() {}

    __device__ void RunLoadRegisterClipboard(const Float* __restrict__ p_src,
                                             Float* __restrict__ p_clipboard) const
    {
    }

    __device__ void RunStoreRegisterClipboard(const Float* __restrict__ p_clipboard,
                                              Float* __restrict__ p_dst) const
    {
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        Float p_clipboard[GetRegisterClipboardSize()];

        RunLoadRegisterClipboard(p_src, p_clipboard);
        RunStoreRegisterClipboard(p_clipboard, p_dst);
    }
};
