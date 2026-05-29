// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sink-aware decode_d128_m16 bf16. See decode_d128_m128 sibling for the
// full doc; this is the "_t" tiny-decode tier (kBlockM=16) used when
// avg_rows ≤ 16 — the canonical q=1 generation step.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(decode_d128_m16, bf16, true, 0, false, true)

} // namespace ck_tile
