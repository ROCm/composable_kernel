// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA prefill_d128 fp8 (IsLocal=true, runtime page_size). The
// per-tensor Q/K/V descales follow the same fold-into-softmax-scale
// path the non-local fp8 instance uses; no extra wiring is needed for
// SWA because the descales are read out of `args` independently of the
// mask logic.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL(prefill_d128, fp8, true, 0, true)

} // namespace ck_tile
