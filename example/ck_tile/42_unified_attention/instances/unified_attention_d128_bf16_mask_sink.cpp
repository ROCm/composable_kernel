// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sink-aware prefill_d128 bf16 (kHasSink=true, runtime page_size, no SWA).
// Picks up the `if constexpr (kHasSink)` init branch in
// `unified_attention_pipeline.hpp` that seeds m / l / o_acc from a
// per-Q-head sink vector. The kernel-side `kargs.sink_ptr` is pre-offset
// by `kv_head_idx * num_queries_per_kv` and forwarded as the pipeline's
// `sink_ptr_pre_offset`.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(prefill_d128, bf16, true, 0, false, true)

} // namespace ck_tile
