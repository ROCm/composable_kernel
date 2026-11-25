// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "ck_tile/core.hpp"

// Value-preserving helpers that steer how LLVM generates code: each returns what plain C++
// would, but in a form that blocks one transformation a hand-scheduled kernel cannot afford.
// Used by MxGemmPipelineAgBgCrCompTDMScheduled and its block GEMM.

namespace ck_tile {
namespace detail {

/// v, opaque to LLVM and held in an SGPR: LLVM can no longer factor it out of the expressions
/// using it (for example fold a scalar stride into each address and keep one VGPR per use).
/// Emits no instruction. v must be wave-uniform.
CK_TILE_DEVICE index_t opaque_scalar(index_t v)
{
#if defined(__gfx12__)
    asm("" : "+s"(v));
#endif
    return v;
}

/// c ? a : b as a bit-mask select. Clang emits ?: on pointers as a branch, LLVM then sinks one
/// side's address math into it and the branch stays divergent; an asm barrier on the address
/// instead reads a VGPR in asm, which under expert scheduling drains the WMMA pipeline.
template <typename T>
CK_TILE_DEVICE const T* select_address(bool c, const T* a, const T* b)
{
    const uint64_t mask = uint64_t{0} - static_cast<uint64_t>(c);
    return reinterpret_cast<const T*>((reinterpret_cast<uint64_t>(a) & mask) |
                                      (reinterpret_cast<uint64_t>(b) & ~mask));
}

} // namespace detail
} // namespace ck_tile
