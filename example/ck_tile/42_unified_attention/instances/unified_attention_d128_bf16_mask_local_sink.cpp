// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA × sink prefill_d128 bf16 (kHasSink=true, IsLocal=true, runtime
// page_size). Combines the SWA Step-D KV-tile clip in the kernel with
// the sink-aware online-softmax init in the pipeline; the two
// `if constexpr` branches are orthogonal and compose. The
// all-window-masked Q-tile case (zero KV overlap) hits the pipeline's
// no-work early-exit and writes lse = sm_scale * sink_raw, o_acc = 0.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(prefill_d128, bf16, true, 0, true, true)

} // namespace ck_tile
