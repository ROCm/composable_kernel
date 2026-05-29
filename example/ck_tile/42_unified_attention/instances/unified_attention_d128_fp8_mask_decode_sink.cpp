// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sink-aware decode_d128_m128 fp8 (kHasSink=true, no SWA). Largest decode
// tile on the d=128 path; reached when GQA fan-out fills an m=128 tile.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(decode_d128_m128, fp8, true, 0, false, true)

} // namespace ck_tile
