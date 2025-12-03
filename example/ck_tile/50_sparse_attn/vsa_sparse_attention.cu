// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "jenga_sparse_attention.h"
#include "fmha_fwd_trek.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/device_memory.hpp"


ck_tile::HostTensor<DataType> vsa_sparse_attention(
    ck_tile::HostTensor<DataType> &TQ,
    ck_tile::HostTensor<DataType> &TK,
    ck_tile::HostTensor<DataType> &TV,
    ck_tile::HostTensor<int32_t> &TKV_block_idx,  // LUT must be int32_t
    ck_tile::HostTensor<int32_t> &TKV_blocks,     // valid_block_num must be int32_t
    ck_tile::HostTensor<DataType> &Y,
    std::optional<ck_tile::HostTensor<DataType>> bias,
    std::optional<ck_tile::HostTensor<DataType>> lse,
    std::optional<ck_tile::HostTensor<DataType>>  seqstart_q,
    std::optional<ck_tile::HostTensor<DataType>>  seqstart_k,
    int bias_type,
    int batch,
    int nhead,
    int nhead_k,
    int seqlen_q,
    int seqlen_k,
    int hdim_q,
    int hdim_v,
    int mode,
    bool i_perm, 
    bool o_perm,
    int max_seqlen_q,
    int max_seqlen_k
){
    std::string data_type = "fp16";
    // DataType is determined at compile time via template

    if (max_seqlen_q == 0) max_seqlen_q = seqlen_q;
    if (max_seqlen_k == 0) max_seqlen_k = seqlen_k;
    bool is_v_rowmajor = true;
    float scale_s = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q));
    float scale_p = 1.f;
    float scale_o = 1.f;
    const float logits_soft_cap = 0.0;

    std::string msk_str = "0";
    mask_info mask = mask_info::decode(msk_str, seqlen_q, seqlen_k);

    const ck_tile::index_t shape_seqlen_q = (mode == 0 ? seqlen_q : max_seqlen_q);
    const ck_tile::index_t shape_seqlen_k = (mode == 0 ? seqlen_k : max_seqlen_k);

    ck_tile::stream_config stream_config{nullptr,
                                         false, // time_kernel
                                         0, /* log_level = */
                                         0,
                                         1,
                                         false};

    // Create device memory and copy data to device
    ck_tile::DeviceMem q_buf(TQ.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(TK.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(TV.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lut_buf(TKV_block_idx.get_element_space_size_in_bytes());
    ck_tile::DeviceMem valid_block_num_buf(TKV_blocks.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(Y.get_element_space_size_in_bytes());

    q_buf.ToDevice(TQ.data());
    k_buf.ToDevice(TK.data());
    v_buf.ToDevice(TV.data());
    lut_buf.ToDevice(TKV_block_idx.data());
    valid_block_num_buf.ToDevice(TKV_blocks.data());

    // Optional buffers
    ck_tile::DeviceMem bias_buf(bias ? bias->get_element_space_size_in_bytes() : 0);
    ck_tile::DeviceMem lse_buf(lse ? lse->get_element_space_size_in_bytes() : 0);
    ck_tile::DeviceMem seqstart_q_buf(seqstart_q ? seqstart_q->get_element_space_size_in_bytes() : 0);
    ck_tile::DeviceMem seqstart_k_buf(seqstart_k ? seqstart_k->get_element_space_size_in_bytes() : 0);

    if (bias) bias_buf.ToDevice(bias->data());
    if (lse) lse_buf.ToDevice(lse->data());
    if (seqstart_q) seqstart_q_buf.ToDevice(seqstart_q->data());
    if (seqstart_k) seqstart_k_buf.ToDevice(seqstart_k->data());

    const auto init_args = [&](auto& args) {
        assert(nhead % nhead_k == 0);
        const ck_tile::index_t stride_q    = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k    = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v    = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return (i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k);
        }();
        const ck_tile::index_t stride_bias    = (i_perm ? max_seqlen_k : 1 * max_seqlen_k);
        const ck_tile::index_t stride_randval = (max_seqlen_k);
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k = i_perm ? shape_seqlen_k * hdim_q : hdim_q;
        const ck_tile::index_t nhead_stride_v    = [&]() {
            if(is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_bias =
            (i_perm ? 0 * shape_seqlen_q * max_seqlen_k : 0 * max_seqlen_k);
        const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_lse     = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k = nhead_k * shape_seqlen_k * hdim_q;
        const ck_tile::index_t batch_stride_v = nhead_k * hdim_v * shape_seqlen_k;
        const ck_tile::index_t batch_stride_bias    = (0 * nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_lse     = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_o     = (nhead * shape_seqlen_q * hdim_v);

        // Use device buffer pointers instead of host tensor data pointers
        args.q_ptr = q_buf.GetDeviceBuffer();
        args.k_ptr = k_buf.GetDeviceBuffer();
        args.v_ptr = v_buf.GetDeviceBuffer();
        args.lut_ptr = lut_buf.GetDeviceBuffer();
        args.valid_block_num_ptr = valid_block_num_buf.GetDeviceBuffer();

        args.batch    = batch;
        args.seqlen_q = shape_seqlen_q; // unused in group mode
        args.hdim_q   = hdim_q;
        args.hdim_v   = hdim_v;
        args.nhead_q  = nhead;
        args.nhead_k  = nhead_k;

        args.stride_q       = stride_q;
        args.stride_k       = stride_k;
        args.stride_v       = stride_v;
        args.nhead_stride_q = nhead_stride_q;
        args.nhead_stride_k = nhead_stride_k;
        args.nhead_stride_v = nhead_stride_v;
        args.batch_stride_q = batch_stride_q;
        args.batch_stride_k = batch_stride_k;
        args.batch_stride_v = batch_stride_v;

        args.bias_ptr = bias ? bias_buf.GetDeviceBuffer() : nullptr;
        args.lse_ptr  = lse ? lse_buf.GetDeviceBuffer() : nullptr;
        args.o_ptr    = o_buf.GetDeviceBuffer();

        args.seqstart_q_ptr = (mode == 1 ? seqstart_q_buf.GetDeviceBuffer() : nullptr);
        args.seqstart_k_ptr = (mode == 1 ? seqstart_k_buf.GetDeviceBuffer() : nullptr);
        args.seqlen_k_ptr =  nullptr;

        args.seqlen_k     = shape_seqlen_k; // unused in group mode (or kvcache enabled)
        args.max_seqlen_q = max_seqlen_q;

        args.scale_s = scale_s;
        args.scale_p = scale_p;
        args.scale_o = scale_o;

        args.logits_soft_cap = logits_soft_cap;

        args.stride_bias =stride_bias;
        args.stride_o          = stride_o;
        args.nhead_stride_bias = nhead_stride_bias;
        args.nhead_stride_lse  = nhead_stride_lse;
        args.nhead_stride_o    = nhead_stride_o;
        args.batch_stride_bias = batch_stride_bias;
        args.batch_stride_lse  = batch_stride_lse;
        args.batch_stride_o    = batch_stride_o;

        args.window_size_left  = mask.left;
        args.window_size_right = mask.right;
        args.mask_type         = static_cast<ck_tile::index_t>(mask.type);

        args.rand_val_ptr = nullptr;

        args.stride_randval       = stride_randval;
        args.nhead_stride_randval = nhead_stride_randval;
        args.batch_stride_randval = batch_stride_randval;

        args.p_drop    = 0.;
        args.s_randval = false;

    };

    const auto init_traits = [&](auto& traits) {
        traits.hdim_q        = hdim_q;
        traits.hdim_v        = hdim_v;
        traits.data_type     = data_type;
        traits.is_v_rowmajor = is_v_rowmajor;


        traits.is_group_mode       = (mode == 1);
        traits.has_logits_soft_cap = 0.f < logits_soft_cap;
        traits.mask_type           = mask.type;
        traits.bias_type           = static_cast<bias_enum>(bias_type);
        traits.has_lse             = lse ? true: false;
        traits.do_fp8_static_quant = false;

        traits.has_dropout = false;

    };

    fmha_jenga_fwd_traits fmha_traits;
    init_traits(fmha_traits);

    fmha_jenga_fwd_args args;
    init_args(args);
    
    fmha_vsa_fwd(fmha_traits, args, stream_config);

    // Copy output back to host
    Y = o_buf.ToHost<DataType>();

    return Y;
}
