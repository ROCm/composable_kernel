// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "jenga_sparse_attention.h"
#include "fmha_fwd_trek.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/device_memory.hpp"
#include <type_traits>

template <typename DataType_>
ck_tile::HostTensor<DataType_>
vsa_sparse_attention(const ck_tile::HostTensor<DataType_>& TQ,
                     const ck_tile::HostTensor<DataType_>& TK,
                     const ck_tile::HostTensor<DataType_>& TV,
                     const ck_tile::HostTensor<int32_t>& TKV_block_idx,
                     const ck_tile::HostTensor<int32_t>& TKV_blocks,
                     ck_tile::HostTensor<DataType_>& Y,
                     int batch,
                     int nhead,
                     int nhead_k,
                     int seqlen_q,
                     int seqlen_k,
                     int hdim_q,
                     int hdim_v,
                     bool i_perm,
                     bool o_perm,
                     int max_seqlen_q,
                     int max_seqlen_k,
                     int log_level)
{
    static_assert(std::is_same_v<DataType_, ck_tile::half_t> ||
                      std::is_same_v<DataType_, ck_tile::bf16_t>,
                  "VSA sparse attention supports fp16/bf16 only.");
    // Determine data type string based on template parameter
    std::string data_type = "fp16";
    if constexpr(std::is_same_v<DataType_, ck_tile::bf16_t>)
    {
        data_type = "bf16";
    }

    if(max_seqlen_q == 0)
        max_seqlen_q = seqlen_q;
    if(max_seqlen_k == 0)
        max_seqlen_k = seqlen_k;
    bool is_v_rowmajor  = true;
    float scale_s       = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q));
    std::string msk_str = "0";
    mask_info mask      = mask_info::decode(msk_str, seqlen_q, seqlen_k);

    const ck_tile::index_t shape_seqlen_q = seqlen_q;
    const ck_tile::index_t shape_seqlen_k = seqlen_k;

    ck_tile::stream_config stream_config{nullptr,
                                         false, // time_kernel
                                         log_level,
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

    const auto init_args = [&](auto& args) {
        assert(nhead % nhead_k == 0);
        const ck_tile::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return (i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k);
        }();
        const ck_tile::index_t stride_o = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k = i_perm ? shape_seqlen_k * hdim_q : hdim_q;
        const ck_tile::index_t nhead_stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_o = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k = nhead_k * shape_seqlen_k * hdim_q;
        const ck_tile::index_t batch_stride_v = nhead_k * hdim_v * shape_seqlen_k;
        const ck_tile::index_t batch_stride_o = (nhead * shape_seqlen_q * hdim_v);

        // Use device buffer pointers instead of host tensor data pointers
        args.q_ptr               = q_buf.GetDeviceBuffer();
        args.k_ptr               = k_buf.GetDeviceBuffer();
        args.v_ptr               = v_buf.GetDeviceBuffer();
        args.lut_ptr             = lut_buf.GetDeviceBuffer();
        args.valid_block_num_ptr = valid_block_num_buf.GetDeviceBuffer();

        args.batch    = batch;
        args.seqlen_q = shape_seqlen_q; // batch mode only
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

        args.o_ptr = o_buf.GetDeviceBuffer();

        args.seqlen_k     = shape_seqlen_k; // batch mode only
        args.max_seqlen_q = max_seqlen_q;

        args.scale_s = scale_s;

        args.stride_o       = stride_o;
        args.nhead_stride_o = nhead_stride_o;
        args.batch_stride_o = batch_stride_o;

        args.window_size_left  = mask.left;
        args.window_size_right = mask.right;
        args.mask_type         = static_cast<ck_tile::index_t>(mask.type);

        // Dropout not supported for sparse attention.
    };

    const auto init_traits = [&](auto& traits) {
        traits.hdim_q        = hdim_q;
        traits.hdim_v        = hdim_v;
        traits.data_type     = data_type;
        traits.is_v_rowmajor = is_v_rowmajor;

        traits.mask_type = mask.type;
    };

    fmha_vsa_fwd_traits fmha_traits;
    init_traits(fmha_traits);

    fmha_vsa_fwd_args args;
    init_args(args);

    fmha_vsa_fwd(fmha_traits, args, stream_config);

    // Copy output back to host without changing tensor shape
    o_buf.FromDevice(Y.data(), Y.get_element_space_size_in_bytes());

    return Y;
}

// Explicit template instantiations
template ck_tile::HostTensor<ck_tile::half_t>
vsa_sparse_attention<ck_tile::half_t>(const ck_tile::HostTensor<ck_tile::half_t>&,
                                      const ck_tile::HostTensor<ck_tile::half_t>&,
                                      const ck_tile::HostTensor<ck_tile::half_t>&,
                                      const ck_tile::HostTensor<int32_t>&,
                                      const ck_tile::HostTensor<int32_t>&,
                                      ck_tile::HostTensor<ck_tile::half_t>&,
                                      int,
                                      int,
                                      int,
                                      int,
                                      int,
                                      int,
                                      int,
                                      bool,
                                      bool,
                                      int,
                                      int,
                                      int);

template ck_tile::HostTensor<ck_tile::bf16_t>
vsa_sparse_attention<ck_tile::bf16_t>(const ck_tile::HostTensor<ck_tile::bf16_t>&,
                                      const ck_tile::HostTensor<ck_tile::bf16_t>&,
                                      const ck_tile::HostTensor<ck_tile::bf16_t>&,
                                      const ck_tile::HostTensor<int32_t>&,
                                      const ck_tile::HostTensor<int32_t>&,
                                      ck_tile::HostTensor<ck_tile::bf16_t>&,
                                      int,
                                      int,
                                      int,
                                      int,
                                      int,
                                      int,
                                      int,
                                      bool,
                                      bool,
                                      int,
                                      int,
                                      int);
