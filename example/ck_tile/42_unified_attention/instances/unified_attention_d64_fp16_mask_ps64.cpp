// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

INST_UNIFIED_ATTENTION_DISPATCH_PS(prefill_d64, fp16, true, 64)

} // namespace ck_tile
