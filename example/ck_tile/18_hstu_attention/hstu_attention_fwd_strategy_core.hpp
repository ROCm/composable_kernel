// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

// Jagged forward tile / split-kv / grid-order strategy: the data and the selection logic, free of
// HIP and CK so unit tests can run on the host. hstu_attention_fwd_strategy.hpp applies it to a
// launch on the current device.

struct HstuFwdConfig
{
    int mtile;
    int num_splits; // 0: no split-kv
    int grid_order; // 1: almost-invariant (batch-leading), 0: length-aware, -1: heuristic
};

struct HstuFwdRange
{
    int lo = 0;
    int hi = INT_MAX;
    bool contains(int v) const { return v >= lo && v <= hi; }
};

struct HstuFwdStrategyRule
{
    // Match; -1 (tri-state) or the default range accepts anything.
    int cross    = -1;
    int causal   = -1;
    int windowed = -1;
    int softmax  = -1;
    HstuFwdRange dtype_bytes, hdim_qk, hdim_v, heads, batch, total_q, total_kv;

    // Action. splits_const >= 0 fixes num_splits (0 disables split-kv); otherwise num_splits fills
    // splits_waves waves of 2 workgroups per CU with Q tiles of mtile rows, rounded to a power of
    // two, clamped to [splits_min, splits_max] (0 below 1.5), and halved while the average
    // sequence has fewer than kv_tiles_per_split 64-wide KV tiles per split.
    int mtile              = 128;
    int grid_order         = -1;
    int splits_const       = -1;
    float splits_waves     = 3.0f;
    int splits_min         = 2;
    int splits_max         = 32;
    int kv_tiles_per_split = 2;
};

struct HstuFwdStrategy
{
    std::string name;
    std::string arch; // empty: any arch
    std::vector<HstuFwdStrategyRule> rules;
};

// What the host knows about a launch without a sync.
struct HstuFwdCallFeatures
{
    bool cross;
    bool causal;
    bool windowed;
    bool softmax;
    int dtype_bytes;
    int hdim_qk;
    int hdim_v;
    int heads;
    int batch;
    int total_q;
    int total_kv;
};

// The rule fitted to recorded production calls swept over every (mtile, split count, grid order)
// config: on gfx950 hdim-128 non-causal cross-attention, mtile 128 and the length-aware grid won
// 107 of 108 calls and about three waves of Q tiles picked the split count.
inline const HstuFwdStrategy& default_hstu_fwd_strategy()
{
    static const HstuFwdStrategy strategy = [] {
        HstuFwdStrategyRule rule;
        rule.cross      = 1;
        rule.causal     = 0;
        rule.windowed   = 0;
        rule.hdim_qk    = {128, 128};
        rule.hdim_v     = {128, 128};
        rule.grid_order = 0;
        return HstuFwdStrategy{"default", "gfx950", {rule}};
    }();
    return strategy;
}

inline bool hstu_fwd_rule_matches(const HstuFwdStrategyRule& r, const HstuFwdCallFeatures& f)
{
    auto tri = [](int want, bool have) { return want < 0 || (want != 0) == have; };
    return tri(r.cross, f.cross) && tri(r.causal, f.causal) && tri(r.windowed, f.windowed) &&
           tri(r.softmax, f.softmax) && r.dtype_bytes.contains(f.dtype_bytes) &&
           r.hdim_qk.contains(f.hdim_qk) && r.hdim_v.contains(f.hdim_v) &&
           r.heads.contains(f.heads) && r.batch.contains(f.batch) &&
           r.total_q.contains(f.total_q) && r.total_kv.contains(f.total_kv);
}

inline int
hstu_fwd_rule_num_splits(const HstuFwdStrategyRule& r, const HstuFwdCallFeatures& f, int num_cu)
{
    if(r.splits_const >= 0)
        return r.splits_const;

    const float q_tiles = static_cast<float>(f.heads) * f.total_q / r.mtile;
    const float wanted  = r.splits_waves * 2.0f * num_cu / q_tiles;
    if(wanted < 1.5f)
        return 0;
    int num_splits = std::clamp(
        1 << static_cast<int>(std::lround(std::log2(wanted))), r.splits_min, r.splits_max);

    const int avg_kv_tiles = f.total_kv / (f.batch * 64);
    while(num_splits > 2 && avg_kv_tiles < r.kv_tiles_per_split * num_splits)
        num_splits /= 2;
    return avg_kv_tiles < r.kv_tiles_per_split * num_splits ? 0 : num_splits;
}

// Returns the index of the rule that picked cfg, or -1 when the strategy does not apply.
inline int select_hstu_fwd_config(const HstuFwdStrategy& strategy,
                                  const HstuFwdCallFeatures& f,
                                  std::string_view arch,
                                  int num_cu,
                                  HstuFwdConfig& cfg)
{
    if(f.total_q <= 0 || f.total_kv <= 0 || f.batch <= 0 ||
       (!strategy.arch.empty() && strategy.arch != arch))
        return -1;
    for(size_t i = 0; i < strategy.rules.size(); ++i)
    {
        const HstuFwdStrategyRule& r = strategy.rules[i];
        if(hstu_fwd_rule_matches(r, f))
        {
            cfg = HstuFwdConfig{r.mtile, hstu_fwd_rule_num_splits(r, f, num_cu), r.grid_order};
            return static_cast<int>(i);
        }
    }
    return -1;
}
