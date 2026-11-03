// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The batch-mode forward traits and arguments that fwd (mha_fwd) and infer (mha_infer) share.

#pragma once

#include "mha_fwd_head_grouping_utils.hpp"

#include <optional>
#include <string>

namespace ck_tile::fmha::torch_api {

inline fmha_fwd_traits get_ck_fmha_fwd_traits(const mask_info& mask,
                                              std::string dtype,
                                              int head_size,
                                              bool has_dropout,
                                              bool has_lse,
                                              bias_enum bias_type)
{
    return fmha_fwd_traits{head_size,
                           head_size,
                           dtype,
                           false, // is_group_mode
                           true,  // is_v_rowmajor
                           false, // has_logits_soft_cap
                           mask.type,
                           bias_type,
                           has_lse,
                           has_dropout,
                           quant_scale_enum::no_scale}; // qscale_type
}

inline fmha_fwd_args get_ck_fmha_fwd_args(bool has_lse,
                                          bool has_dropout_randval,
                                          const mask_info& mask,
                                          // sizes
                                          const int b,
                                          const int seqlen_q,
                                          const int seqlen_k,
                                          const int h,
                                          const int h_k,
                                          const int d,
                                          // device pointers
                                          const at::Tensor q,
                                          const at::Tensor k,
                                          const at::Tensor v,
                                          const std::optional<at::Tensor>& alibi_slopes_,
                                          const std::optional<at::Tensor>& bias_,
                                          at::Tensor out,
                                          at::Tensor softmax_lse,
                                          at::Tensor dropout_randval,
                                          float softmax_scale,
                                          float p_dropout,
                                          std::pair<uint64_t*, uint64_t*> drop_seed_offset)
{
    // q: (batch_size, seqlen_q, nheads, d)
    // k: (batch_size, seqlen_k, nheads_k, d)
    // v: (batch_size, seqlen_k, nheads_k, d)
    // o: (batch_size, seqlen_q, nheads, d)

    // alibi_slopes:(batch_size, nheads) or (nhead)
    // lse: (batch_size, nheads, seqlen_q)
    // randval: (batch_size, nheads, seqlen_q, seqlen_k)

    ck_tile::index_t stride_q       = q.stride(1);
    ck_tile::index_t stride_k       = k.stride(1);
    ck_tile::index_t stride_v       = v.stride(1);
    ck_tile::index_t stride_o       = out.stride(1);
    ck_tile::index_t stride_randval = has_dropout_randval ? dropout_randval.stride(2) : 0;

    ck_tile::index_t nhead_stride_q       = q.stride(2);
    ck_tile::index_t nhead_stride_k       = k.stride(2);
    ck_tile::index_t nhead_stride_v       = v.stride(2);
    ck_tile::index_t nhead_stride_o       = out.stride(2);
    ck_tile::index_t nhead_stride_lse     = has_lse ? softmax_lse.stride(1) : 0;
    ck_tile::index_t nhead_stride_randval = has_dropout_randval ? dropout_randval.stride(1) : 0;

    ck_tile::index_t batch_stride_q = q.stride(0);
    ck_tile::index_t batch_stride_k = k.stride(0);
    ck_tile::index_t batch_stride_v = v.stride(0);
    ck_tile::index_t batch_stride_o = out.stride(0);

    ck_tile::index_t batch_stride_lse     = has_lse ? softmax_lse.stride(0) : 0;
    ck_tile::index_t batch_stride_randval = has_dropout_randval ? dropout_randval.stride(0) : 0;

    void* bias_ptr               = nullptr;
    ck_tile::index_t stride_bias = 0;

    if(bias_.has_value())
    {
        bias_ptr    = bias_.value().data_ptr();
        stride_bias = bias_.value().stride(0);
    }
    else if(alibi_slopes_.has_value())
    {
        const auto& alibi_slopes = alibi_slopes_.value();
        CK_TILE_FMHA_CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1,
                    "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) ||
                        alibi_slopes.sizes() == torch::IntArrayRef({b, h}),
                    "ALiBi slopes must be [nheads] or [batch_size, nheads]");
        bias_ptr    = alibi_slopes.data_ptr();
        stride_bias = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    return fmha_fwd_args{nullptr, // selected_kernel_name
                         q.data_ptr(),
                         k.data_ptr(),
                         v.data_ptr(),
                         bias_ptr, // bias
                         nullptr,  // q_descale_ptr
                         nullptr,  // k_descale_ptr
                         nullptr,  // v_descale_ptr
                         has_dropout_randval ? dropout_randval.data_ptr() : nullptr,
                         has_lse ? softmax_lse.data_ptr() : nullptr,
                         out.data_ptr(),
                         nullptr, // seqstart_q_ptr
                         nullptr, // seqstart_k_ptr
                         nullptr, // seqlen_q_ptr
                         nullptr, // seqlen_k_ptr
                         nullptr, // cu_seqlen_q_ptr
                         nullptr, // cu_seqlen_k_ptr
                         nullptr, // block_scale_seqstart_q_ptr
                         nullptr, // block_scale_seqstart_k_ptr
                         nullptr, // seqstart_v_scale_ptr
                         nullptr, // sink_ptr
                         seqlen_q,
                         seqlen_k,
                         b,
                         seqlen_q,      // max_seqlen_q
                         d,             // hdim_q
                         d,             // hdim_v
                         h,             // nhead
                         h_k,           // nhead_k
                         0,             // num_head_q_total
                         0,             // head_start
                         softmax_scale, // scale_s
                         0.0f,          // logits_soft_cap
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_bias,
                         stride_randval,
                         stride_o,
                         0, // stride_q_descale
                         0, // stride_k_descale
                         0, // stride_v_descale
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         0, // nhead_stride_bias: broadcast across heads
                         nhead_stride_randval,
                         nhead_stride_lse,
                         nhead_stride_o,
                         0, // nhead_stride_q_descale
                         0, // nhead_stride_k_descale
                         0, // nhead_stride_v_descale
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         0, // batch_stride_bias: broadcast across batches
                         batch_stride_randval,
                         batch_stride_lse,
                         batch_stride_o,
                         0, // batch_stride_q_descale
                         0, // batch_stride_k_descale
                         0, // batch_stride_v_descale
                         mask.left,
                         mask.right,
                         0, // sink_size
                         static_cast<ck_tile::index_t>(mask.type),
                         0, // min_seqlen_q
                         p_dropout,
                         has_dropout_randval,
                         drop_seed_offset,
                         0,  // block_scale_size_q
                         0}; // block_scale_size_kv
}

} // namespace ck_tile::fmha::torch_api
