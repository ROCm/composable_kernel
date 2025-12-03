#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
#include <optional>
#include <cstdint>
#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

using DataType = ck_tile::half_t;

ck_tile::HostTensor<DataType> jenga_sparse_attention(
    ck_tile::HostTensor<DataType> &TQ,
    ck_tile::HostTensor<DataType> &TK,
    ck_tile::HostTensor<DataType> &TV,
    ck_tile::HostTensor<DataType> &Tblock_relation_onehot,
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
);

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
);
