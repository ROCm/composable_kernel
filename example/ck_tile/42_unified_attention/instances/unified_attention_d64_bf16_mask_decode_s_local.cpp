// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA decode_d64_m64 bf16 (IsLocal=true, runtime page_size). Fills the
// middle tier of the d=64 decode SWA ladder — relevant primarily for
// num_qpkv=1 (MHA) workloads where the dispatcher's `avg_rows = avg_q *
// num_qpkv` lands in (16, 64].
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL(decode_d64_m64, bf16, true, 0, true)

} // namespace ck_tile
