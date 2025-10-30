// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <utility>

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host/stream_config.hpp"

namespace ck_tile {

struct fmha_fwd_v3_args
{
    enum class data_type_enum
    {
        fp16,
        bf16
    };

    data_type_enum data_type;
    // bool is_varlen;

    index_t batch;
    index_t seqlen_q;
    index_t seqlen_k;
    index_t nhead_q;
    index_t nhead_kv;
    index_t hdim_qk;
    index_t hdim_v;

    float softmax_scale;

    index_t window_size_left;
    index_t window_size_right;
    index_t mask_type; // should be 0 for no mask; or 2 for causal mask (window_size_left < 0 and
                       // window_size_right == 0).

    const void* q_ptr;
    index_t stride_q;
    index_t nhead_stride_q;
    index_t batch_stride_q;

    const void* k_ptr;
    index_t stride_k;
    index_t nhead_stride_k;
    index_t batch_stride_k;

    const void* v_ptr;
    index_t stride_v;
    index_t nhead_stride_v;
    index_t batch_stride_v;

    void* o_ptr;
    index_t stride_o;
    index_t nhead_stride_o;
    index_t batch_stride_o;

    // Optional batch-mode cumulative seqlen overrides (exclude PAD)
    // If provided, they override per-batch effective lengths to skip tail padding.
    const ck_tile::index_t* cu_seqlen_q_ptr  = nullptr; // [batch+1]
    const ck_tile::index_t* cu_seqlen_kv_ptr = nullptr; // [batch+1]
};

std::ostream& operator<<(std::ostream& stream, const fmha_fwd_v3_args::data_type_enum& data_type);

// return value:
//   first  = whether the kernel was launched (true = launched, false = skipped)
//   second = elapsed time (ms) of the kernel launch, valid only if first == true
std::pair<bool, float> fmha_fwd_v3(const fmha_fwd_v3_args& args, const stream_config& config);

} // namespace ck_tile
