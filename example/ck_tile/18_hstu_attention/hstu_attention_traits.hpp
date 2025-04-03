// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadSeqLenQ_,
          bool kPadSeqLenK_,
          bool kPadHeadDimQK_,
          bool kPadHeadDimV_,
          index_t kBlockPerCu_>
struct HstuAttentionFwdTraits
{
    static constexpr bool kPadSeqLenQ   = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK   = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQK = kPadHeadDimQK_;
    static constexpr bool kPadHeadDimV  = kPadHeadDimV_;

    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

} // namespace ck_tile
