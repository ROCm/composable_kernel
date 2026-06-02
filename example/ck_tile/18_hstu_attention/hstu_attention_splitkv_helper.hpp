// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "hstu_attention_util.hpp"

static float get_estimated_cu_coverage_ratio(int num_batches, int num_heads, int max_seqlen_q)
{
    int num_CUs  = get_number_of_cu();
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    int nbatch_nhead_mblocks = num_batches * num_heads * ceildiv(max_seqlen_q, 64);

    // assume each CU can run two work-groups, common cases for hdim128
    return static_cast<float>(nbatch_nhead_mblocks) / (2.0f * num_CUs);
};

static bool shall_use_splitkv(int num_batches, int num_heads, int max_seqlen_q)
{
    // Please tune the threshold here
    const float threshold = 0.8f;

    if(get_estimated_cu_coverage_ratio(num_batches, num_heads, max_seqlen_q) < threshold)
        return true;
    return false;
};

static int get_suggested_num_splits(int num_batches, int num_heads, int max_seqlen_q)
{
    int i = 2;

    // Please tune the threshold here
    const float threshold = 1.5f;
    while(get_estimated_cu_coverage_ratio(num_batches, num_heads, max_seqlen_q) * i < threshold)
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
