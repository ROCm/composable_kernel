// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sliding-Window-Attention decode_d64_m16 bf16 — the q=1 tiny tier. This
// is the primary GPT-OSS decode shape (single-token generation step with
// a 4k-128k context window).
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL(decode_d64_m16, bf16, true, 0, true)

} // namespace ck_tile
