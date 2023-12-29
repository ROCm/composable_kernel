// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/type.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename BiasDataType_,
          typename PDataType_,
          typename OaccDataType_,
          typename ODataType_,
          index_t kBlockSize_,
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
    using PDataType           = remove_cvref_t<PDataType_>;
    using OaccDataType        = remove_cvref_t<OaccDataType_>;
    using ODataType           = remove_cvref_t<ODataType_>;
    using BlockFmhaShape      = remove_cvref_t<BlockFmhaShape_>;
    using FmhaMask            = remove_cvref_t<FmhaMask_>;
    using Traits              = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = kBlockSize_;
    static constexpr bool kIsGroupMode  = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kM0NeedPadding   = Traits::kM0NeedPadding;
    static constexpr bool kN0K1NeedPadding = Traits::kN0K1NeedPadding;
    static constexpr bool kHasBias         = Traits::kHasBias;
    static constexpr index_t kBlockPerCu   = Traits::kBlockPerCu;
};

} // namespace block
} // namespace tile_program
} // namespace ck
