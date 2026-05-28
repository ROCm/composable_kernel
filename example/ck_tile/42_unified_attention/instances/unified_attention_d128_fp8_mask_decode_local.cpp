// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA decode_d128_m128 fp8 (IsLocal=true, runtime page_size).
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL(decode_d128_m128, fp8, true, 0, true)

} // namespace ck_tile
