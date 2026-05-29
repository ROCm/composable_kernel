// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA × sink prefill_d64 bf16. See d128 local_sink sibling for the full
// doc.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(prefill_d64, bf16, true, 0, true, true)

} // namespace ck_tile
