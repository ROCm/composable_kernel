// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/block/block_rotary_embedding.hpp"

namespace ck_tile {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename BlockSageAttnShape_,
          bool kIsGroupMode_,
          typename AttentionVariant_,
          typename AttnMask_,
          typename Traits_>
struct BlockSageAttnPipelineProblem
{
    using QDataType           = remove_cvref_t<QDataType_>;
    using KDataType           = remove_cvref_t<KDataType_>;
    using VDataType           = remove_cvref_t<VDataType_>;
    using SaccDataType        = remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType = remove_cvref_t<SMPLComputeDataType_>;
    using PDataType           = remove_cvref_t<PDataType_>;
    using OaccDataType        = remove_cvref_t<OaccDataType_>;
    using ODataType           = remove_cvref_t<ODataType_>;
    using BlockSageAttnShape  = remove_cvref_t<BlockSageAttnShape_>;
    using AttentionVariant    = remove_cvref_t<AttentionVariant_>;
    using AttnMask            = remove_cvref_t<AttnMask_>;
    using Traits              = remove_cvref_t<Traits_>;

    static constexpr index_t kNumGemm0Warps = BlockSageAttnShape::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = BlockSageAttnShape::NumGemm1Warps;
    static constexpr index_t kBlockSize     = BlockSageAttnShape::NumWarps * get_warp_size();

    static constexpr bool kIsGroupMode = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ     = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK     = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ    = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV    = Traits::kPadHeadDimV;
    static constexpr bool kSkipMinSeqlenQ = Traits::kSkipMinSeqlenQ;
    static constexpr auto QScaleEnum      = Traits::QScaleEnum;
    static constexpr index_t kBlockPerCu  = Traits::kBlockPerCu;

    /// Must match host scale tensor layout (same values as TileSageAttnTraits for Sage kernels).
    static constexpr index_t kBlockScaleSizeQ = Traits::kBlockScaleSizeQ;
    static constexpr index_t kBlockScaleSizeK = Traits::kBlockScaleSizeK;
};

} // namespace ck_tile
