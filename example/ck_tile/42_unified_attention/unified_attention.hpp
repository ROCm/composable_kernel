// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <utility>

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/ops/unified_attention.hpp"

namespace ck_tile {

struct unified_attention_args
{
    enum class data_type_enum
    {
        fp16,
        bf16,
        fp8 // e4m3fn on gfx950 (OCP) / e4m3fnuz on gfx942 (FNUZ); selected by CK_TILE_USE_OCP_FP8
    };

    data_type_enum data_type;
    // bool is_varlen;
    index_t mask_type; // should be 0 for no mask; or 2 for causal mask (window_size_left < 0 and
                       // window_size_right == 0).

    index_t num_tokens; // total number of tokens in query
    index_t num_blks;
    index_t num_head_q;
    index_t num_queries_per_kv;
    index_t page_blk_size;
    // index_t BLOCK_SIZE;

    index_t hdim;
    // TODO window
    float scale_s; // softmax scale (1/sqrt(d) by convention); pre-multiplied with log2(e)
                   // inside MakeKargs so the device-side softmax can use exp2.
    // Per-tensor FP8 descales (a.k.a. "scales" in the Triton kernel naming). All three
    // default to 1.0f so non-FP8 dtypes round-trip cleanly. q_descale and k_descale are
    // folded into scale_s inside the pipeline (so the softmax sees the combined scalar);
    // v_descale is applied once to o_acc after the 1/l normalization, outside the K/V
    // loop. Matches Triton unified_attention's q_scale/k_scale/v_scale semantics
    // (see aiter/ops/triton/_triton_kernels/attention/unified_attention.py:110-114, 351-358).
    float q_descale = 1.0f;
    float k_descale = 1.0f;
    float v_descale = 1.0f;
    // Legacy fields kept for ABI stability with downstream callers (csrc glue, examples).
    // The pipeline currently uses q_descale/k_descale/v_descale instead.
    float scale     = 1.0f;
    float scale_k   = 1.0f;
    float scale_v   = 1.0f;
    float scale_out = 1.0f;

    const void* q_ptr;
    index_t query_stride_0;
    index_t query_stride_1;

    const void* k_ptr; // [num_blks, blk_size, num_kv_heads, head_size]
    index_t stride_k_cache_0;
    index_t stride_k_cache_1;
    index_t stride_k_cache_2;
    index_t stride_k_cache_3;

    const void* v_ptr; // [num_blks, blk_size, num_kv_heads, head_size]
    index_t stride_v_cache_0;
    index_t stride_v_cache_1;
    index_t stride_v_cache_2;
    index_t stride_v_cache_3;

    void* o_ptr;
    index_t output_stride_0;
    index_t output_stride_1;

    const int32_t* block_tables_ptr;
    index_t block_table_stride;
    const int32_t* seq_lens_ptr;        // seq len in each batch
    const int32_t* query_start_len_ptr; // [num_seqs+1]

    index_t num_seqs; // number of batches for q
    index_t max_seqlen_q = 0; // max query length across all batches (0 = unknown)

    // Layout selector for K/V:
    //   false (default) : paged KV cache — K/V are
    //                     [num_blks, page_blk_size, num_kv_heads, head_size]
    //                     and `block_tables_ptr` resolves logical → physical
    //                     page. Used by vLLM / SGLang inference servers.
    //   true            : contiguous (THD) — K/V are
    //                     [num_kv_tokens, num_kv_heads, head_size] for the
    //                     current request and `block_tables_ptr` is ignored.
    //                     The kernel skips the per-tile page-table fetch and
    //                     the / % page_size arithmetic entirely. Used by
    //                     pretraining / flash-attention-style callers that
    //                     don't have a shared KV cache. When this is true,
    //                     `page_blk_size` is ignored; treat `num_blks` as a
    //                     virtual page count: the K/V tensor view still has
    //                     shape (num_blks * page_blk_size, head_dim), so the
    //                     caller can either set num_blks=num_kv_tokens with
    //                     page_blk_size=1 or any equivalent factorisation
    //                     that yields the right `num_kv_tokens` total.
    bool kv_contiguous = false;

    // Set to true when the K/V cache is large enough that an int32 byte
    // offset into it can overflow (i.e. when
    //   num_blocks * page_size * num_kv_heads * head_dim * sizeof(T) > INT32_MAX
    // ). When true, the pipeline routes K/V async loads through
    // `global_load_lds` (per-lane 64-bit base ptr); when false, it uses the
    // faster `buffer_load_dword_lds` path with a shared 4 GB-capped SRD.
    bool cache_ptr_int32_overflow_possible = false;

    // KV-segment parallelism (split-KV). When num_splits == 1, the kernel
    // writes to o_ptr as usual. When num_splits > 1, the kernel is launched
    // with a 3D grid whose z-dim is num_splits — each CTA computes its own
    // partial (o_acc, lse) and writes them into the FP32 workspaces; a
    // separate combine kernel (or a Python-side reduce) merges across
    // splits into the final output.
    //
    // Workspace layout (host-allocated):
    //   o_acc_ptr   : [num_q_heads, num_splits, total_q, hdim_v]  (FP32)
    //   lse_acc_ptr : [num_q_heads, num_splits, total_q]          (FP32)
    // The corresponding host-set strides are below.
    index_t num_splits = 1;
    void* o_acc_ptr    = nullptr;
    void* lse_acc_ptr  = nullptr;
    index_t split_stride_o_acc    = 0;
    index_t split_stride_lse_acc  = 0;
    index_t nhead_stride_o_acc    = 0;
    index_t nhead_stride_lse_acc  = 0;
};

std::ostream& operator<<(std::ostream& stream,
                         const unified_attention_args::data_type_enum& data_type);

// return value:
//   first  = whether the kernel was launched (true = launched, false = skipped)
//   second = elapsed time (ms) of the kernel launch, valid only if first == true
std::pair<bool, float> unified_attention(const unified_attention_args& args,
                                         const stream_config& config);

} // namespace ck_tile

struct UnifiedAttentionMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};
