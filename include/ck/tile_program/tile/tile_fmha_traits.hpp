// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tile_program {

template <bool kM0NeedPadding_ /* padding for seqlen_q */,
          bool kN0K1NeedPadding_ /* padding for seqlen_k */,
          bool kHasBias_,
          index_t kBlockPerCu_ = 2 /* hint to occupancy */>
struct TileFmhaTraits
{
    static constexpr bool kM0NeedPadding   = kM0NeedPadding_;
    static constexpr bool kN0K1NeedPadding = kN0K1NeedPadding_;
    static constexpr bool kHasBias         = kHasBias_;
    static constexpr index_t kBlockPerCu   = kBlockPerCu_;
};

} // namespace tile_program
} // namespace ck
