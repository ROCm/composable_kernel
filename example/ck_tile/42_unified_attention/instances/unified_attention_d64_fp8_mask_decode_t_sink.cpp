// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sink-aware decode_d64_m16 fp8 (kHasSink=true, no SWA). The canonical
// GPT-OSS generation-step tile: d=64 GQA-8 with q=1 (avg_rows = 8 ≤ 16).
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(decode_d64_m16, fp8, true, 0, false, true)

} // namespace ck_tile
