// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadHeadDim_ /* paddding for hdim_v */,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */,
          bool kHasSink_       = false /* learnable per-Q-head attention sink */>
struct TileUnifiedAttentionTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadHeadDim    = kPadHeadDim_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
    // When true, the pipeline reads a per-Q-head sink scalar at init
    // time and seeds the online softmax with the corresponding virtual
    // key (GPT-OSS / vLLM convention). The kernel forwards the pointer
    // via `kargs.sink_ptr`. Default `false` reproduces the classic
    // no-sink softmax; no instance flips this yet.
    static constexpr bool kHasSink = kHasSink_;
};
} // namespace ck_tile
