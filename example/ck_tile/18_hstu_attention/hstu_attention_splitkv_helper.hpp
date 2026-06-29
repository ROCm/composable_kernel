// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/math.hpp>

#include "hstu_attention_host_util.hpp"

static int
get_hstu_attention_fwd_mtile(int num_batches, int num_heads, int max_seqlen_q, int max_seqlen_kv)
{
    int num_CUs  = get_number_of_cu();
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    if(max_seqlen_q <= 64)
        return 64;

    // for cross-attention where max_seqlen_kv is much bigger than max_seqlen_q, we always use
    // mtile_size 128, not to worry about the CU coverage, since split-kv can help us to solve
    if(max_seqlen_q >= 128 && static_cast<float>(max_seqlen_kv) / max_seqlen_q >= 5.0)
        return 128;

    int nbatch_nhead_mblocks = num_batches * num_heads * ceildiv(max_seqlen_q, 128);

    // assuming each CU is assigned two work-groups
    if(nbatch_nhead_mblocks >= static_cast<int>(0.85f * num_CUs * 2.0f))
        return 128;

    // currently, only hdim-128 actually uses mtile-64, for other hdim, the settings for
    // mtile-64 can be added through tuning/verification
    return 64;
};

static float
get_estimated_cu_coverage_ratio(int num_batches, int num_heads, int max_seqlen_q, int max_seqlen_kv)
{
    int num_CUs  = get_number_of_cu();
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    int m_tile_size =
        get_hstu_attention_fwd_mtile(num_batches, num_heads, max_seqlen_q, max_seqlen_kv);

    int nbatch_nhead_mblocks = num_batches * num_heads * ceildiv(max_seqlen_q, m_tile_size);

    // assume each CU can run two work-groups, common cases for hdim128
    return static_cast<float>(nbatch_nhead_mblocks) / (2.0f * num_CUs);
};

static bool shall_use_splitkv(int num_batches, int num_heads, int max_seqlen_q, int max_seqlen_kv)
{
    // Please tune the threshold here
    const float threshold = (max_seqlen_kv >= 2048) ? 1.5f : 0.8f;

    if(get_estimated_cu_coverage_ratio(num_batches, num_heads, max_seqlen_q, max_seqlen_kv) <
       threshold)
        return true;
    return false;
};

static int
get_suggested_num_splits(int num_batches, int num_heads, int max_seqlen_q, int max_seqlen_kv)
{
    int i = 2;

    // Please tune the threshold here
    const float threshold = 3.0f;
    while(get_estimated_cu_coverage_ratio(num_batches, num_heads, max_seqlen_q, max_seqlen_kv) * i <
          threshold)
        i++;

    // the num_splits shall not be bigger than 64
    return ck_tile::min(i, 64);
};

struct SplitkvWorkspace
{
    int num_splits;
    void* o_acc_ptr;
    void* lse_acc_ptr; // only used when softmax is used
};
