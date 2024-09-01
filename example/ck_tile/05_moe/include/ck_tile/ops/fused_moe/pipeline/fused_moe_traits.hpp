// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moe_weight_permute_enum.hpp"

namespace ck_tile {

template <bool DownPreShuffled_ = false,
          index_t kBlockPerCu_  = -1 /* overwrite occupancy if not -1 */,
          index_t OAtomic_      = 0,
          FusedMoeWeightPermuteEnum WeightPermute_ =
              FusedMoeWeightPermuteEnum::permute_b_nr_kr_kw_nw_kv>
struct FusedMoeTraits
{
    static constexpr bool DownPreShuffled                    = DownPreShuffled_;
    static constexpr index_t kBlockPerCu                     = kBlockPerCu_;
    static constexpr FusedMoeWeightPermuteEnum WeightPermute = WeightPermute_;
    static constexpr index_t OAtomic = OAtomic_; // 0-pack fp16/bf16 atomic, 1-fp32 atomic
};
} // namespace ck_tile
