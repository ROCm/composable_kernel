// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// Sink-aware decode_d64_m64 bf16 (the d=64 "_s" tier; kBlockM=64).
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(decode_d64_m64, bf16, true, 0, false, true)

} // namespace ck_tile
