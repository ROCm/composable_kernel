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

enum class FusedMoePermuteStyle
{
    // permute_b_n0_k0_n1_k1_n2_k2 = 0, // 0,1,4,2,5,3,6
    // permute_b_n0_n1_k0_k1_n2_k2 = 1, // 0,1,2,4,5,3,6
    permute_b_nr_kr_kw_nw_kv    = 2, // 0,1,3,4,2,5
    permute_b_nr_kr_waveflatten = permute_b_nr_kr_kw_nw_kv,
    no_permute                  = 999,
};

template <bool DownPreShuffled_              = false,
          FusedMoePermuteStyle PermuteStyle_ = FusedMoePermuteStyle::permute_b_nr_kr_kw_nw_kv,
          index_t kBlockPerCu_               = -1 /* overwrite occupancy if not -1 */>
struct FusedMoeTraits
{
    static constexpr bool DownPreShuffled              = DownPreShuffled_;
    static constexpr FusedMoePermuteStyle PermuteStyle = PermuteStyle_;
    static constexpr index_t kBlockPerCu               = kBlockPerCu_;
};
} // namespace ck_tile
