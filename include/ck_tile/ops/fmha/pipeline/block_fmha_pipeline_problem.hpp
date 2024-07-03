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
          typename LSEDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          typename Traits_>
struct BlockFmhaPipelineProblem
{
    using QDataType             = remove_cvref_t<QDataType_>;
    using KDataType             = remove_cvref_t<KDataType_>;
    using VDataType             = remove_cvref_t<VDataType_>;
    using SaccDataType          = remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType   = remove_cvref_t<SMPLComputeDataType_>;
    using BiasDataType          = remove_cvref_t<BiasDataType_>;
    using RandValOutputDataType = remove_cvref_t<RandValOutputDataType_>;
    using LSEDataType           = remove_cvref_t<LSEDataType_>;
    using PDataType             = remove_cvref_t<PDataType_>;
    using OaccDataType          = remove_cvref_t<OaccDataType_>;
    using ODataType             = remove_cvref_t<ODataType_>;
    using BlockFmhaShape        = remove_cvref_t<BlockFmhaShape_>;
    using FmhaMask              = remove_cvref_t<FmhaMask_>;
    using Traits                = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = BlockFmhaShape::NumWarps * get_warp_size();
    static constexpr bool kIsGroupMode  = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ       = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = Traits::kPadHeadDimV;
    static constexpr auto BiasEnum          = Traits::BiasEnum;
    static constexpr bool kStoreLSE         = Traits::kStoreLSE;
    static constexpr bool kHasDropout       = Traits::kHasDropout;
    static constexpr bool kDoFp8StaticQuant = Traits::kDoFp8StaticQuant;
    static constexpr index_t kBlockPerCu    = Traits::kBlockPerCu;
};

template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename SaccDataType,
          typename SMPLComputeDataType,
          typename BiasDataType,
          typename RandValOutputDataType,
          typename LSEDataType,
          typename PDataType,
          typename OaccDataType,
          typename ODataType,
          typename BlockFmhaShape,
          bool kIsGroupMode,
          typename FmhaMask,
          typename Traits>
struct BlockFmhaFwdSplitKVPipelineProblem : BlockFmhaPipelineProblem<QDataType,
                                                                     KDataType,
                                                                     VDataType,
                                                                     SaccDataType,
                                                                     SMPLComputeDataType,
                                                                     BiasDataType,
                                                                     RandValOutputDataType,
                                                                     LSEDataType,
                                                                     PDataType,
                                                                     OaccDataType,
                                                                     ODataType,
                                                                     BlockFmhaShape,
                                                                     kIsGroupMode,
                                                                     FmhaMask,
                                                                     Traits>
{
    static constexpr bool kHasUnevenSplits = kIsGroupMode || Traits::kHasUnevenSplits;
};

template <typename LSEDataType_,
          typename OaccDataType_,
          typename ODataType_,
          index_t HeadDimV_,
          index_t kM0_,
          index_t kN1_,
          bool kIsGroupMode_,
          typename Traits_>
struct BlockFmhaSplitKVCombinePipelineProblem
{
    using LSEDataType  = remove_cvref_t<LSEDataType_>;
    using OaccDataType = remove_cvref_t<OaccDataType_>;
    using ODataType    = remove_cvref_t<ODataType_>;
    using Traits       = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = 256;
    static constexpr bool kIsGroupMode  = kIsGroupMode_;

    static constexpr index_t kHeadDimV = HeadDimV_;
    static constexpr index_t kM0       = kM0_;
    static constexpr index_t kN1       = kN1_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ       = Traits::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV      = Traits::kPadHeadDimV;
    static constexpr bool kStoreLSE         = Traits::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant = Traits::kDoFp8StaticQuant;
    static constexpr index_t kBlockPerCu    = Traits::kBlockPerCu;
    static constexpr index_t kMaxSplits     = Traits::kMaxSplits;
};

} // namespace ck_tile
