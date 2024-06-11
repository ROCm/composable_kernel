// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"

namespace ck_tile {

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          bool kStoreLSE_,
          bool kHasDropout_,
          bool kDoFp8StaticQuant_,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */,
          index_t kMaxSplits_  = 1>
struct TileFmhaTraits
{
    static constexpr bool kPadSeqLenQ       = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK       = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQ      = kPadHeadDimQ_;
    static constexpr bool kPadHeadDimV      = kPadHeadDimV_;
    static constexpr auto BiasEnum          = BiasEnum_;
    static constexpr bool kHasBiasGrad      = kHasBiasGrad_;
    static constexpr bool kStoreLSE         = kStoreLSE_;
    static constexpr bool kHasDropout       = kHasDropout_;
    static constexpr bool kDoFp8StaticQuant = kDoFp8StaticQuant_;
    static constexpr index_t kBlockPerCu    = kBlockPerCu_;
    static constexpr index_t kMaxSplits     = kMaxSplits_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          index_t kBlockPerCu_ = 2 /* hint to occupancy */>
struct TileFmhaBwdOGradDotOTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadHeadDimV   = kPadHeadDimV_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

} // namespace ck_tile
