// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename BiasDataType_,
          typename RandValOutputDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename UnifiedAttentionShape_,
          typename FmhaMask_,
          typename Traits_>
struct UnifiedAttentionPipelineProblem
{
    // TODO kM0 and KN1??
    using QDataType = remove_cvref_t<QDataType_>;
    using KDataType = remove_cvref_t<KDataType_>;
    using VDataType = remove_cvref_t<VDataType_>;
    // first gemm accumulation dtype
    using SaccDataType = remove_cvref_t<SaccDataType_>;
    // Softmax dtype
    using SMPLComputeDataType   = remove_cvref_t<SMPLComputeDataType_>;
    using BiasDataType          = remove_cvref_t<BiasDataType_>;
    using RandValOutputDataType = remove_cvref_t<RandValOutputDataType_>;
    // data type for A matrix of second gemm
    using PDataType = remove_cvref_t<PDataType_>;
    // data type for second gemm accumulation
    using OaccDataType          = remove_cvref_t<OaccDataType_>;
    using ODataType             = remove_cvref_t<ODataType_>;
    using UnifiedAttentionShape = remove_cvref_t<UnifiedAttentionShape_>;
    using Traits                = remove_cvref_t<Traits_>;
    using FmhaMask              = remove_cvref_t<FmhaMask_>;

    static constexpr index_t kNumGemm0Warps = UnifiedAttentionShape::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = UnifiedAttentionShape::NumGemm1Warps;
    static constexpr index_t kBlockSize     = UnifiedAttentionShape::NumWarps * get_warp_size();

    // attributes from traits
    static constexpr bool kPadSeqLenQ       = Traits::kPadSeqLenQ;
    static constexpr bool kPadHeadDim       = Traits::kPadHeadDim;
    static constexpr bool kHasLogitsSoftCap = Traits::kHasLogitsSoftCap;
    static constexpr bool kSkipMinSeqlenQ   = Traits::kSkipMinSeqlenQ;
    static constexpr bool kHasDropout       = Traits::kHasDropout;
    static constexpr bool kDoFp8StaticQuant = Traits::kDoFp8StaticQuant;
    static constexpr index_t kBlockPerCu    = Traits::kBlockPerCu;
};
} // namespace ck_tile
