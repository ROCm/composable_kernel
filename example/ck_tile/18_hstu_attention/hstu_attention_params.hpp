// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

struct HstuAttentionFwdParams
{
    bool is_jagged;

    ck_tile::index_t num_batch;
    ck_tile::index_t seqlen;     // batched mode only
    const void* seq_offsets_ptr; // jagged mode only
    ck_tile::index_t max_seqlen; // jagged mode only

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    void* o_ptr;

    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;
    ck_tile::index_t num_head;
    float scale_s;

    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_bias;
    ck_tile::index_t seq_stride_o;

    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_o;

    // batched mode only parameters
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_o;

    const void* num_targets_ptr;

    bool use_causal;
    ck_tile::index_t window_size;
    ck_tile::index_t contextual_seqlen;
    ck_tile::index_t min_full_attn_seqlen;

    float p_drop;
    uint64_t philox_seed;
    uint64_t philox_offset;
};
