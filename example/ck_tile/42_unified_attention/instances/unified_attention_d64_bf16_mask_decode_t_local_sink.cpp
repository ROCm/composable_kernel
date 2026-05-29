// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA × sink decode_d64_m16 bf16 — the canonical GPT-OSS-with-sink call
// (q=1 generation, GQA-8, windowed). Combines the SWA Step-D KV-tile
// clip with the sink-aware online-softmax init.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(decode_d64_m16, bf16, true, 0, true, true)

} // namespace ck_tile
