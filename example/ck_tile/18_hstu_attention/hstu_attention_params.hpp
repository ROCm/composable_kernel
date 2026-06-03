// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

struct HstuAttentionNoGroupFwdParams
{
    // for self-attention (is_cross_attention = false), we requires
    // 1) either seqlen_kv == 0 or seqlen_kv == seqlen_q
    // 2) either seq_kv_offsets_ptr == nullptr, or seq_kv_offsets_ptr == seq_q_offsets_ptr
    bool is_cross_attention;

    bool is_jagged;

    bool use_softmax;

    bool is_training;

    ck_tile::index_t num_batch;
    ck_tile::index_t seqlen_q;      // batched mode only
    ck_tile::index_t seqlen_kv;     // batched mode only
    const void* seq_q_offsets_ptr;  // jagged mode only
    const void* seq_kv_offsets_ptr; // jagged mode only
    ck_tile::index_t max_seqlen_q;  // jagged mode only

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    void* o_ptr;
    void* lse_ptr; // only used when both is_training and use_softmax be true

    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;
    ck_tile::index_t num_head;
    float scale_s;    // scaling factor exerted on the immediate Q@K result
    float attn_scale; // scaling factor exerted on the SiLU result

    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_bias;
    ck_tile::index_t seq_stride_o;
    ck_tile::index_t seq_stride_lse;

    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_lse;

    // batched mode only parameters
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t batch_stride_lse;

    const void* num_targets_ptr;

    bool use_causal;
    // parameters used by Non-Group HSTU
    ck_tile::index_t window_size;
    ck_tile::index_t contextual_seqlen;
    ck_tile::index_t min_full_attn_seqlen;

    float p_drop;
    uint64_t philox_seed;
    uint64_t philox_offset;
};

struct HstuAttentionGroupFwdParams
{
    // for self-attention (is_cross_attention = false), we requires
    // 1) either seq_kv_offsets_ptr == nullptr, or seq_kv_offsets_ptr == seq_q_offsets_ptr
    bool is_cross_attention;

    bool use_softmax;

    bool is_training;

    ck_tile::index_t num_group;
    ck_tile::index_t num_batch;
    const void* seq_q_offsets_ptr;
    const void* seq_kv_offsets_ptr;
    ck_tile::index_t max_seqlen_q; // the maximum of all the groups' max_seqlen_q

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    void* o_ptr;
    void* lse_ptr; // only used when both is_training and use_softmax be true

    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;
    ck_tile::index_t num_head;
    float scale_s; // scaling factor exerted on the immediate Q@K result

    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_bias;
    ck_tile::index_t seq_stride_o;
    ck_tile::index_t seq_stride_lse;

    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_lse;

    const void* num_targets_ptr;

    bool use_causal;

    // parameters used by Group HSTU
    const void* group_attn_scale_ptr;
    const void* group_max_seqlen_q_ptr; // use for setting attn_scales
    const void* group_window_size_ptr;
    const void* group_contextual_seqlen_ptr;
    const void* group_min_full_attn_seqlen_ptr;

    float p_drop;
    uint64_t philox_seed;
    uint64_t philox_offset;
};
