// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          bool kHasBias_,
          bool kStoreLSE_,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK    = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQ   = kPadHeadDimQ_;
    static constexpr bool kPadHeadDimV   = kPadHeadDimV_;
    static constexpr bool kHasBias       = kHasBias_;
    static constexpr bool kStoreLSE      = kStoreLSE_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

} // namespace ck_tile
