// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sink-aware prefill_d128 fp8 (kHasSink=true, runtime page_size, no SWA).
// fp8 sibling of the bf16/fp16 prefill_d128 sink instance.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(prefill_d128, fp8, true, 0, false, true)

} // namespace ck_tile
