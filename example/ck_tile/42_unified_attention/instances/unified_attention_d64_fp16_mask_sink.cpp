// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sink-aware prefill_d64 fp16 (kHasSink=true, runtime page_size, no SWA).
// See `unified_attention_d128_bf16_mask_sink.cpp` for the full doc.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(prefill_d64, fp16, true, 0, false, true)

} // namespace ck_tile
