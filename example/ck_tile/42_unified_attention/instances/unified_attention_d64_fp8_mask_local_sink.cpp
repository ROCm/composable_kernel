// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA × sink prefill_d64 fp8 (IsLocal=true, kHasSink=true). Picks up the
// SWA Step-D clip in the kernel and the sink init in the pipeline; the
// two `if constexpr` branches are orthogonal and compose without extra
// code.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(prefill_d64, fp8, true, 0, true, true)

} // namespace ck_tile
