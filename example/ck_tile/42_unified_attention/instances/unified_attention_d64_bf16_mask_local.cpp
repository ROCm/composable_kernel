// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sliding-Window-Attention prefill_d64 bf16 (IsLocal=true, runtime page_size).
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL(prefill_d64, bf16, true, 0, true)

} // namespace ck_tile
