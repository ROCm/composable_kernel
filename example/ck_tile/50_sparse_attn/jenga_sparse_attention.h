// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <optional>
#include <cstdint>
#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

template <typename DataType_>
ck_tile::HostTensor<DataType_>
jenga_sparse_attention(const ck_tile::HostTensor<DataType_>& TQ,
                       const ck_tile::HostTensor<DataType_>& TK,
                       const ck_tile::HostTensor<DataType_>& TV,
                       const ck_tile::HostTensor<uint8_t>& Tblock_relation_onehot,
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
                       int log_level = 0);

template <typename DataType_>
ck_tile::HostTensor<DataType_> vsa_sparse_attention(
    const ck_tile::HostTensor<DataType_>& TQ,
    const ck_tile::HostTensor<DataType_>& TK,
    const ck_tile::HostTensor<DataType_>& TV,
    const ck_tile::HostTensor<int32_t>& TKV_block_idx, // LUT must be int32_t
    const ck_tile::HostTensor<int32_t>& TKV_blocks,    // valid_block_num must be int32_t
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
    int log_level = 0);
