// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "sparge_tool.hpp"

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
                                   int log_level = 0);
