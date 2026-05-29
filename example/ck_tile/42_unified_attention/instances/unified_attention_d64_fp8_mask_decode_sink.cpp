// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sink-aware decode_d64_m128 fp8 (kHasSink=true, no SWA). Reached when GQA
// fans out enough rows to fill an m=128 tile (e.g. nqpkv=8 with
// avg_q ∈ [9, 16]).
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(decode_d64_m128, fp8, true, 0, false, true)

} // namespace ck_tile
