// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA × sink decode_d64_m16 fp8 (IsLocal=true, kHasSink=true). The
// canonical GPT-OSS-with-sink call: d=64 GQA-8 generation step (q=1)
// with sliding window + sink, on the fp8 weight path.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL_SINK(decode_d64_m16, fp8, true, 0, true, true)

} // namespace ck_tile
