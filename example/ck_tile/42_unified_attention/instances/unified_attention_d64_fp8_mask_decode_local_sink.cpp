// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA × sink decode_d64_m128 fp8 (IsLocal=true, kHasSink=true).
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(decode_d64_m128, fp8, true, 0, true, true)

} // namespace ck_tile
