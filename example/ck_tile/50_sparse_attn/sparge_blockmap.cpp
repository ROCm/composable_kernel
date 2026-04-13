// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "sparge_blockmap.h"
#include "sparge_blockmap_trek.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/device_memory.hpp"
#include <type_traits>
#include <cmath>

template <typename DataType_>
sparge::VSALut sparge_blockmap_gpu(const ck_tile::HostTensor<DataType_>& TQ,
                                   const ck_tile::HostTensor<DataType_>& TK,
                                   ck_tile::HostTensor<uint8_t>& block_map_out,
                                   int batch,
                                   int nhead_q,
                                   int nhead_k,
                                   int seqlen_q,
                                   int seqlen_k,
                                   int hdim_q,
                                   bool i_perm,
                                   float simthreshd1,
                                   float cdfthreshd,
                                   float topk,
                                   int blkq,
                                   int blkk,
                                   int log_level)
{
    static_assert(std::is_same_v<DataType_, ck_tile::half_t> ||
                      std::is_same_v<DataType_, ck_tile::bf16_t>,
                  "sparge_blockmap_gpu supports fp16/bf16 only.");

    std::string data_type = "fp16";
    if constexpr(std::is_same_v<DataType_, ck_tile::bf16_t>)
    {
        data_type = "bf16";
    }

    const ck_tile::index_t num_q_blocks = ck_tile::integer_divide_ceil(seqlen_q, blkq);
    const ck_tile::index_t num_k_blocks = ck_tile::integer_divide_ceil(seqlen_k, blkk);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim_q));

    // Allocate device memory
    ck_tile::DeviceMem q_buf(TQ.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(TK.get_element_space_size_in_bytes());

    const std::size_t bmap_bytes =
        static_cast<std::size_t>(batch) * nhead_q * num_q_blocks * num_k_blocks * sizeof(uint8_t);
    const std::size_t lut_bytes =
        static_cast<std::size_t>(batch) * nhead_q * num_q_blocks * num_k_blocks * sizeof(int32_t);
    const std::size_t valid_bytes =
        static_cast<std::size_t>(batch) * nhead_q * num_q_blocks * sizeof(int32_t);

    ck_tile::DeviceMem bmap_buf(bmap_bytes);
    ck_tile::DeviceMem lut_buf(lut_bytes);
    ck_tile::DeviceMem valid_buf(valid_bytes);

    q_buf.ToDevice(TQ.data());
    k_buf.ToDevice(TK.data());
    bmap_buf.SetZero();
    lut_buf.SetZero();
    valid_buf.SetZero();

    // Compute strides (assumes BHSD if i_perm, BSHD otherwise)
    const ck_tile::index_t stride_q = i_perm ? hdim_q : nhead_q * hdim_q;
    const ck_tile::index_t stride_k = i_perm ? hdim_q : nhead_k * hdim_q;
    const ck_tile::index_t nhead_stride_q =
        i_perm ? static_cast<ck_tile::index_t>(seqlen_q) * hdim_q : hdim_q;
    const ck_tile::index_t nhead_stride_k =
        i_perm ? static_cast<ck_tile::index_t>(seqlen_k) * hdim_q : hdim_q;
    const ck_tile::index_t batch_stride_q =
        static_cast<ck_tile::index_t>(nhead_q) * seqlen_q * hdim_q;
    const ck_tile::index_t batch_stride_k =
        static_cast<ck_tile::index_t>(nhead_k) * seqlen_k * hdim_q;

    ck_tile::stream_config stream_config{nullptr, false, log_level, 0, 1, false};

    sparge_blockmap_args args;
    args.q_ptr               = q_buf.GetDeviceBuffer();
    args.k_ptr               = k_buf.GetDeviceBuffer();
    args.batch               = batch;
    args.seqlen_q            = seqlen_q;
    args.seqlen_k            = seqlen_k;
    args.hdim_q              = hdim_q;
    args.nhead_q             = nhead_q;
    args.nhead_k             = nhead_k;
    args.stride_q            = stride_q;
    args.stride_k            = stride_k;
    args.nhead_stride_q      = nhead_stride_q;
    args.nhead_stride_k      = nhead_stride_k;
    args.batch_stride_q      = batch_stride_q;
    args.batch_stride_k      = batch_stride_k;
    args.simthreshd1         = simthreshd1;
    args.cdfthreshd          = cdfthreshd;
    args.topk                = topk;
    args.scale               = scale;
    args.block_map_ptr       = bmap_buf.GetDeviceBuffer();
    args.lut_ptr             = lut_buf.GetDeviceBuffer();
    args.valid_block_num_ptr = valid_buf.GetDeviceBuffer();

    sparge_blockmap_traits traits;
    traits.data_type = data_type;
    traits.hdim_q    = hdim_q;

    sparge_blockmap_fwd(traits, args, stream_config);

    // Copy results back to host
    bmap_buf.FromDevice(block_map_out.data(), bmap_bytes);

    sparge::VSALut vsa_lut{
        ck_tile::HostTensor<int32_t>({batch, nhead_q, num_q_blocks, num_k_blocks}),
        ck_tile::HostTensor<int32_t>({batch, nhead_q, num_q_blocks}),
    };
    lut_buf.FromDevice(vsa_lut.lut.data(), lut_bytes);
    valid_buf.FromDevice(vsa_lut.valid_block_num.data(), valid_bytes);

    return vsa_lut;
}

// Explicit template instantiations
template sparge::VSALut
sparge_blockmap_gpu<ck_tile::half_t>(const ck_tile::HostTensor<ck_tile::half_t>&,
                                     const ck_tile::HostTensor<ck_tile::half_t>&,
                                     ck_tile::HostTensor<uint8_t>&,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     bool,
                                     float,
                                     float,
                                     float,
                                     int,
                                     int,
                                     int);

template sparge::VSALut
sparge_blockmap_gpu<ck_tile::bf16_t>(const ck_tile::HostTensor<ck_tile::bf16_t>&,
                                     const ck_tile::HostTensor<ck_tile::bf16_t>&,
                                     ck_tile::HostTensor<uint8_t>&,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     bool,
                                     float,
                                     float,
                                     float,
                                     int,
                                     int,
                                     int);
