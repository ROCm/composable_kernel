// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"

namespace ck_tile {

// SWA decode_d128_m16 fp8 (IsLocal=true, runtime page_size). m16 uses
// the 16x16x32 MFMA shape on fp8; the LDS-backed P-tile re-layout in
// unified_attention_pipeline.hpp covers it identically to the larger
// tiers (block-store with QK-C distribution, sync, block-load with
// PV-A distribution) — see the fp8 nmask_decode_t comment for the
// per-shape analysis.
INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL(decode_d128_m16, fp8, true, 0, true)

} // namespace ck_tile
