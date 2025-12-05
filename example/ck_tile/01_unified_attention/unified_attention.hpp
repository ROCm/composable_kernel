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
        bf16
    };

    data_type_enum data_type;
    // bool is_varlen;
    index_t mask_type; // should be 0 for no mask; or 2 for causal mask (window_size_left < 0 and
                       // window_size_right == 0).

    index_t num_tokens; // total number of tokens in query
    index_t num_blks;
    index_t num_head_q;
    index_t num_queries_per_kv;
    index_t BLOCK_SIZE;

    index_t hdim;
    // TODO window
    float scale_s;
    float scale_q;
    float scale_k;
    float scale_v;
    float scale_out;
    index_t fp8_mode;

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
