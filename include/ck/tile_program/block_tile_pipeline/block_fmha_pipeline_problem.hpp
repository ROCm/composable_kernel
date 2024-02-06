// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/get_id.hpp"
#include "ck/utility/type.hpp"
#include "ck/utility/data_type.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename BiasDataType_,
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
    using QDataType           = remove_cvref_t<QDataType_>;
    using KDataType           = remove_cvref_t<KDataType_>;
    using VDataType           = remove_cvref_t<VDataType_>;
    using SaccDataType        = remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType = remove_cvref_t<SMPLComputeDataType_>;
    using BiasDataType        = remove_cvref_t<BiasDataType_>;
    using LSEDataType         = remove_cvref_t<LSEDataType_>;
    using PDataType           = remove_cvref_t<PDataType_>;
    using OaccDataType        = remove_cvref_t<OaccDataType_>;
    using ODataType           = remove_cvref_t<ODataType_>;
    using BlockFmhaShape      = remove_cvref_t<BlockFmhaShape_>;
    using FmhaMask            = remove_cvref_t<FmhaMask_>;
    using Traits              = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = BlockFmhaShape::NumWarps * get_warp_size();
    static constexpr bool kIsGroupMode  = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ    = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK    = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ   = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV   = Traits::kPadHeadDimV;
    static constexpr bool kHasBias       = Traits::kHasBias;
    static constexpr bool kStoreLSE      = Traits::kStoreLSE;
    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
    static constexpr bool kIsFp8 =
        (is_same_v<QDataType, f8_t> || is_same_v<QDataType, bf8_t>)&&(
            is_same_v<KDataType, f8_t> ||
            is_same_v<KDataType, bf8_t>)&&(is_same_v<VDataType, f8_t> ||
                                           is_same_v<VDataType, bf8_t>)&&is_same_v<SaccDataType,
                                                                                   float> &&
        is_same_v<OaccDataType, float>;
};

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename GemmDataType_,
          typename LSEDataType_,
          typename AccDataType_,
          typename DDataType_,
          typename ZDataType_,
          typename BiasDataType_,
          typename ODataType_,
          typename OGradDataType_,
          typename QGradDataType_,
          typename KGradDataType_,
          typename VGradDataType_,
          typename BiasGradDataType_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          typename Traits_>
struct BlockFmhaBwdPipelineProblem
{
    using QDataType        = remove_cvref_t<QDataType_>;
    using KDataType        = remove_cvref_t<KDataType_>;
    using VDataType        = remove_cvref_t<VDataType_>;
    using GemmDataType     = remove_cvref_t<GemmDataType_>;
    using LSEDataType      = remove_cvref_t<LSEDataType_>;
    using AccDataType      = remove_cvref_t<AccDataType_>;
    using DDataType        = remove_cvref_t<DDataType_>;
    using ZDataType        = remove_cvref_t<ZDataType_>;
    using BiasDataType     = remove_cvref_t<BiasDataType_>;
    using ODataType        = remove_cvref_t<ODataType_>;
    using OGradDataType    = remove_cvref_t<OGradDataType_>;
    using QGradDataType    = remove_cvref_t<QGradDataType_>;
    using KGradDataType    = remove_cvref_t<KGradDataType_>;
    using VGradDataType    = remove_cvref_t<VGradDataType_>;
    using BiasGradDataType = remove_cvref_t<BiasGradDataType_>;
    using BlockFmhaShape   = remove_cvref_t<BlockFmhaShape_>;
    using FmhaMask         = remove_cvref_t<FmhaMask_>;
    using Traits           = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = BlockFmhaShape::NumWarps * get_warp_size();
    static constexpr bool kIsGroupMode  = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ    = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK    = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ   = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV   = Traits::kPadHeadDimV;
    static constexpr bool kHasBias       = Traits::kHasBias;
    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
};

template <typename ODataType_,
          typename OGradDataType_,
          typename DDataType_,
          index_t kBlockSize_,
          index_t kVHeaddim_,
          bool kIsGroupMode_,
          typename Traits_>
struct BlockFmhaBwdOGradDotOPipelineProblem
{
    using ODataType     = remove_cvref_t<ODataType_>;
    using OGradDataType = remove_cvref_t<OGradDataType_>;
    using DDataType     = remove_cvref_t<DDataType_>;
    using Traits        = remove_cvref_t<Traits_>;

    static_assert(0 < kBlockSize_ && kBlockSize_ % get_warp_size() == 0,
                  "kBlockSize should be divisible by get_warp_size()");

    static constexpr index_t kBlockSize = kBlockSize_;
    static constexpr index_t kVHeaddim  = kVHeaddim_;
    static constexpr bool kIsGroupMode  = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ    = Traits::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV   = Traits::kPadHeadDimV;
    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
};

} // namespace block
} // namespace tile_program
} // namespace ck
