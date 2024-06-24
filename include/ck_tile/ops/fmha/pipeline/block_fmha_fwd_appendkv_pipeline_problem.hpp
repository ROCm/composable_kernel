// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          typename Traits_>
struct BlockFmhaFwdAppendKVPipelineProblem
{
    using QDataType      = remove_cvref_t<QDataType_>;
    using KDataType      = remove_cvref_t<KDataType_>;
    using VDataType      = remove_cvref_t<VDataType_>;
    using BlockFmhaShape = remove_cvref_t<BlockFmhaShape_>;
    using Traits         = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = BlockFmhaShape::NumWarps * get_warp_size();
    static constexpr bool kIsGroupMode  = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ    = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK    = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ   = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV   = Traits::kPadHeadDimV;
    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
};

} // namespace ck_tile
