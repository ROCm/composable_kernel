// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/amd_buffer_coherence.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

/// global_prefetch of the cache line holding addr (global memory) with the given cache policy,
/// on gfx1250; a no-op elsewhere. Unlike buffer_view's GlobalPrefetchDataOp it puts no compiler
/// fences around the prefetch, so the caller's schedule decides what may move across it.
template <amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void amd_global_prefetch([[maybe_unused]] const void* addr)
{
#if defined(__gfx125__)
    __builtin_amdgcn_global_prefetch(addr, static_cast<index_t>(Coherence));
#endif
}

} // namespace ck_tile
