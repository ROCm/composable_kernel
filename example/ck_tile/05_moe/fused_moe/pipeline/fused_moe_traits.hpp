// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_default_policy.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <bool GateUpPreShuffled_ = false,
          bool DownPreShuffled_   = false,
          index_t NumPrefetchA_   = 2,
          index_t NumPrefetchG_   = 2,
          index_t NumPrefetchU_   = 2,
          index_t NumPrefetchD_   = 2,
          index_t kBlockPerCu_    = -1 /* overwrite occupancy if not -1 */>
struct FusedMoeTraits
{
    static constexpr bool GateUpPreShuffled = GateUpPreShuffled_;
    static constexpr bool DownPreShuffled   = DownPreShuffled_;
    static constexpr index_t NumPrefetchA   = NumPrefetchA_;
    static constexpr index_t NumPrefetchG   = NumPrefetchG_;
    static constexpr index_t NumPrefetchU   = NumPrefetchU_;
    static constexpr index_t NumPrefetchD   = NumPrefetchD_;
    static constexpr index_t kBlockPerCu    = kBlockPerCu_;
};
} // namespace ck_tile
