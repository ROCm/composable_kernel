// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

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
    static constexpr bool kM0NeedPadding   = Traits::kM0NeedPadding;
    static constexpr bool kN0K1NeedPadding = Traits::kN0K1NeedPadding;
    static constexpr bool kK0N1NeedPadding = Traits::kK0N1NeedPadding;
    static constexpr bool kHasBias         = Traits::kHasBias;
    static constexpr bool kStoreLSE        = Traits::kStoreLSE;
    static constexpr index_t kBlockPerCu   = Traits::kBlockPerCu;
    static constexpr bool kIsFp8 =
        (is_same_v<QDataType, f8_t> || is_same_v<QDataType, bf8_t>)&&(
            is_same_v<KDataType, f8_t> ||
            is_same_v<KDataType, bf8_t>)&&(is_same_v<VDataType, f8_t> ||
                                           is_same_v<VDataType, bf8_t>)&&is_same_v<SaccDataType,
                                                                                   float> &&
        is_same_v<OaccDataType, float>;
};

} // namespace block
} // namespace tile_program
} // namespace ck
