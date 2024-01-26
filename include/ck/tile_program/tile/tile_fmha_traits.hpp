// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tile_program {

template <bool kM0NeedPadding_ /* padding for seqlen_q */,
          bool kN0K1NeedPadding_ /* padding for seqlen_k */,
          bool kK0N1NeedPadding_ /* paddding for hdim_q / hdim_v */,
          bool kHasBias_,
          bool kStoreLSE_,
          bool kHasDropout_,
          index_t kBlockPerCu_ = 2 /* hint to occupancy */>
struct TileFmhaTraits
{
    static constexpr bool kM0NeedPadding   = kM0NeedPadding_;
    static constexpr bool kN0K1NeedPadding = kN0K1NeedPadding_;
    static constexpr bool kK0N1NeedPadding = kK0N1NeedPadding_;
    static constexpr bool kHasBias         = kHasBias_;
    static constexpr bool kStoreLSE        = kStoreLSE_;
    static constexpr bool kHasDropout      = kHasDropout_;
    static constexpr index_t kBlockPerCu   = kBlockPerCu_;
};

} // namespace tile_program
} // namespace ck
