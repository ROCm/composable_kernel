// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sliding-Window-Attention decode_d64_m128 bf16 (IsLocal=true, runtime
// page_size). Pairs with the existing INST_UNIFIED_ATTENTION_DISPATCH
// `_mask_decode` instance — same variant, same dtype, just the IsLocal=
// true cousin that honours both window bounds.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL(decode_d64_m128, bf16, true, 0, true)

} // namespace ck_tile
