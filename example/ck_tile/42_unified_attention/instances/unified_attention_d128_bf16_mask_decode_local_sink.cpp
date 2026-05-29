// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA × sink decode_d128_m128 bf16. See prefill_d128 local_sink sibling
// for the full doc; this is the m128 decode tier (avg_rows ≤ 128).
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(decode_d128_m128, bf16, true, 0, true, true)

} // namespace ck_tile
