// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA prefill_d64 fp8 (IsLocal=true, runtime page_size).
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL(prefill_d64, fp8, true, 0, true)

} // namespace ck_tile
