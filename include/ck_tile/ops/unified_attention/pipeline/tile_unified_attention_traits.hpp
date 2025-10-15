// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_rotary_embedding.hpp"

namespace ck_tile {


template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadHeadDim /* paddding for hdim_v */,
          bool kStoreLSE_,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileUnifiedAttentionTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadHeadDim   = kPadHeadDim;
    static constexpr bool kStoreLSE      = kStoreLSE_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};
}