// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

// KV cache memory layout selector.
//
// Layout summary (kVectorSize = 16 / sizeof(KDataType)):
// - VECTORIZED_LAYOUT (swizzled):
//   K: [NumBlocks, NumHeads, HeadDim/kVectorSize, PageSize, kVectorSize]
//   V: [NumBlocks, NumHeads, PageSize/kVectorSize, HeadDim, kVectorSize]
// - LINEAR_LAYOUT:
//   K: [NumBlocks, PageSize, NumHeads, HeadDim]
//   V: [NumBlocks, PageSize, NumHeads, HeadDim]
// - LINEAR_HEADS_FIRST_LAYOUT (cross-layer 5D KV cache, non-contiguous view):
//   K: [NumBlocks, NumHeads, PageSize, HeadDim]
//   V: [NumBlocks, NumHeads, PageSize, HeadDim]
//   The view originates from a 6D physical buffer
//   (NumBlocks, NumHeads, NumLayers, 2, PageSize, HeadDim) sliced per layer; the address
//   arithmetic is identical to LINEAR_LAYOUT but with `nhead_stride_k`, `batch_stride_k`,
//   and `stride_k` reflecting the cross-layer permutation. The kernel treats it as
//   LINEAR_LAYOUT at dispatch time; only the AITER wrapper distinguishes them when
//   extracting strides from the input tensor.
enum class BlockAttentionKVCacheMemoryLayoutEnum
{
    VECTORIZED_LAYOUT         = 0,
    LINEAR_LAYOUT             = 1,
    LINEAR_HEADS_FIRST_LAYOUT = 2,
};

// KV cache lookup table layout selector.
// - VLLM_BLOCK_TABLE_2D: block_table[batch, max_blocks_per_seq]
// - SGLANG_PAGE_TABLE_1D: kv_page_indices[kv_indptr[b] ... kv_indptr[b+1])
enum class BlockAttentionKVCacheLookupTableEnum
{
    VLLM_BLOCK_TABLE_2D  = 0,
    SGLANG_PAGE_TABLE_1D = 1,
};

} // namespace ck_tile
