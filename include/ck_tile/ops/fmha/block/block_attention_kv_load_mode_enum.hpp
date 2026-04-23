// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

// KV cache load addressing mode selector for batch_prefill / paged-attention pipelines.
// - BUFFER_LOAD:     SGPR-based SRD via buffer_load_* (default; 32-bit byte addressing, <2GB pool)
// - GLOBAL_LOAD_LDS: direct global_load_lds_* (64-bit addressing, required for >2GB KV cache)
enum class BlockAttentionKVCacheLoadModeEnum
{
    BUFFER_LOAD     = 0,
    GLOBAL_LOAD_LDS = 1,
};

} // namespace ck_tile
