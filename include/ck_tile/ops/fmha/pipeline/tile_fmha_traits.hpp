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
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
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
};

template <bool kPadSeqLenQ /* padding for seqlen_q */,
          bool kPadSeqLenK /* padding for seqlen_k */,
          bool kPadHeadDimQ /* paddding for hdim_q */,
          bool kPadHeadDimV /* paddding for hdim_v */,
          BlockAttentionBiasEnum BiasEnum,
          bool kHasBiasGrad,
          bool kStoreLSE,
          bool kHasDropout,
          bool kDoFp8StaticQuant,
          bool kHasUnevenSplits_ = true,
          index_t kBlockPerCu    = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaFwdSplitKVTraits : TileFmhaTraits<kPadSeqLenQ,
                                                 kPadSeqLenK,
                                                 kPadHeadDimQ,
                                                 kPadHeadDimV,
                                                 BiasEnum,
                                                 kHasBiasGrad,
                                                 kStoreLSE,
                                                 kHasDropout,
                                                 kDoFp8StaticQuant,
                                                 kBlockPerCu>
{
    // determine if some split (length) is not divisible by tile size
    static constexpr bool kHasUnevenSplits = kHasUnevenSplits_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          bool kStoreLSE_,
          bool kDoFp8StaticQuant_,
          index_t kLogMaxSplits_,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaFwdSplitKVCombineTraits
{
    static constexpr bool kPadSeqLenQ       = kPadSeqLenQ_;
    static constexpr bool kPadHeadDimV      = kPadHeadDimV_;
    static constexpr bool kStoreLSE         = kStoreLSE_;
    static constexpr bool kDoFp8StaticQuant = kDoFp8StaticQuant_;

    static constexpr index_t kMaxSplits = (1 << kLogMaxSplits_);
    static_assert(kMaxSplits <= get_warp_size() || kMaxSplits % get_warp_size() == 0);
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
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
