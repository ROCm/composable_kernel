// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sink-aware decode_d128_m128 bf16 (kHasSink=true, runtime page_size, no SWA).
// Mirrors the prefill sink instance from earlier in the rollout; routes
// here when the variant selector lands on `decode_d128_m128` and the
// caller passed a non-null sink vector. Compiles `if constexpr (kHasSink)`
// on, picking up the per-row `m = sink_raw / sm_scale` init in the
// pipeline.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(decode_d128_m128, bf16, true, 0, false, true)

} // namespace ck_tile
