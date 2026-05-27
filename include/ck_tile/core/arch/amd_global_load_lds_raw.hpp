// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

#include "ck_tile/core/arch/amd_buffer_coherence.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"

// =============================================================================
// global_load_lds path — direct DRAM->LDS load via per-lane 64-bit base pointer.
//
// Standalone helper used by `tile_scatter_gather::async_load_raw_long` (and,
// indirectly, by ck_tile's unified_attention op for >2 GB-cache decode/prefill).
// Lives in its own file to keep additions to the long-standing
// `amd_buffer_addressing[_builtins].hpp` HW-utility headers minimal — the
// function below doesn't exist upstream and is needed only by the new op.
//
// Equivalent of `amd_async_buffer_load_with_oob_raw` but bypasses the SRD
// (`int32x4_t` / `__amdgpu_buffer_rsrc_t` resource descriptor) entirely. The
// buffer_load path has two 32-bit limits at the HW boundary:
//   - SRD `size` field is uint32_t (max ~4 GB pool). Caches above that wrap.
//   - `buffer_load_*` voffset is 32-bit. Per-lane offsets above 4 GB wrap.
// Replacing the underlying HW instruction with `global_load_lds` (per-lane
// 64-bit VGPR-pair base + 13-bit signed immediate offset) lifts both limits.
// Required for paged-KV caches whose
// `num_blocks * page_size * row_stride * sizeof(T)` exceeds INT32_MAX (e.g.
// very-long-context decode pools).
//
// Caveats:
//   - Loses the SRD's free OOB clamp. Caller must ensure the per-lane pointer
//     is valid (in our pipeline use, the page_table lookup guarantees this).
//   - gfx9.4+ / gfx950 only — uses `__builtin_amdgcn_global_load_lds`. Older
//     arches would need a `global_load + ds_write` fallback.
//
// Toolchain note (`CK_TILE_HAS_GLOBAL_LOAD_LDS_DWORDX4_BUILTIN`):
//   The size=12/size=16 ImmArg overloads of `__builtin_amdgcn_global_load_lds`
//   for gfx950 only landed in AMD clang ~21+ (verified absent in ROCm 7.1.1
//   / clang 20, present in ROCm 7.11.0 / clang 22). On older toolchains the
//   front-end rejects the size literal at parse time — no flag fixes this.
//   The macro below gates on `__clang_major__ >= 21`; when 0 we fall back to
//   emitting `global_load_lds_dwordx{1,3,4}` via inline asm, which bypasses
//   the ImmArg check entirely and produces the exact same HW instruction
//   (verified zero perf delta vs. the builtin path across the decode
//   regression suite). Override the heuristic manually with
//   `-DCK_TILE_HAS_GLOBAL_LOAD_LDS_DWORDX4_BUILTIN=0/1`.
// =============================================================================

#ifndef CK_TILE_HAS_GLOBAL_LOAD_LDS_DWORDX4_BUILTIN
#if __clang_major__ >= 21
#define CK_TILE_HAS_GLOBAL_LOAD_LDS_DWORDX4_BUILTIN 1
#else
#define CK_TILE_HAS_GLOBAL_LOAD_LDS_DWORDX4_BUILTIN 0
#endif
#endif

namespace ck_tile {

template <typename T,
          index_t N,
          index_t byte_offset_imm             = 0, // 13-bit signed
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_async_global_load_lds_raw(T* smem,
                                                  const T* base_ptr_64,
                                                  bool_constant<pre_nop> = {})
{
    constexpr index_t bytes = sizeof(T) * N;

    static_assert(bytes == 4 || bytes == 12 || bytes == 16,
                  "global_load_lds: only dword / dwordx3 / dwordx4 supported on gfx950");
    static_assert(-4096 <= byte_offset_imm && byte_offset_imm <= 4095,
                  "global_load_lds: byte_offset_imm must fit in 13-bit signed");

    // C-style cast injects the address-space attribute the intrinsic expects
    // (addrspace(1) for global, addrspace(3) for LDS) without losing const.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    const __attribute__((address_space(1))) void* gptr =
        (const __attribute__((address_space(1))) void*)base_ptr_64;
    __attribute__((address_space(3))) void* lptr =
        (__attribute__((address_space(3))) void*)smem;
#pragma clang diagnostic pop

    if constexpr(pre_nop)
        asm volatile("s_nop 4\n" ::: "memory");

    // Front-end requires `size`, `offset` and `aux` to be ImmArg / integer
    // literals. A switch on the constexpr `bytes` value lets each branch
    // pass the literal directly.
    constexpr int kCoherence = static_cast<int>(coherence);
#if CK_TILE_HAS_GLOBAL_LOAD_LDS_DWORDX4_BUILTIN
    if constexpr(bytes == 16)
        __builtin_amdgcn_global_load_lds(gptr, lptr, 16, byte_offset_imm, kCoherence);
    else if constexpr(bytes == 12)
        __builtin_amdgcn_global_load_lds(gptr, lptr, 12, byte_offset_imm, kCoherence);
    else /* bytes == 4 */
        __builtin_amdgcn_global_load_lds(gptr, lptr, 4, byte_offset_imm, kCoherence);
#else
    // Old-toolchain fallback (ROCm ≤ 7.1.1 / AMD clang ≤ 20).
    //
    // The size=12/16 ImmArg overloads of `__builtin_amdgcn_global_load_lds`
    // are rejected during semantic analysis on these compilers, so we emit
    // the dwordx{1,3,4} instruction via inline asm instead — the assembler
    // happily accepts the mnemonic and stamps an identical HW instruction
    // to the one the newer builtin would lower to. (Decomposing into N×
    // size=4 builtin calls *looks* equivalent but isn't: the in-LDS layout
    // of a native `dwordx4` doesn't reduce to any combination of dword
    // INST.OFFSET steps we could find that survives all decode shapes —
    // observed FAIL on b=128 / sk=16384 / d=128 / bf16. Easier to just
    // ask the assembler for the real instruction.)
    //
    // Operand contract:
    //   - M0 (LDS dest base): set explicitly by us via `s_mov_b32`. The
    //     addrspace(3) `lptr` narrows to a 32-bit LDS byte offset on cast.
    //     `readfirstlane` guarantees the value lands in an SGPR even if
    //     LLVM lost sight of its wave-uniformity. The "s" constraints
    //     enforce SALU placement; `m0_dep` plumbs an SSA edge between the
    //     m0 setter and the load asm so LLVM cannot reorder the two.
    //   - `gptr` (per-lane 64-bit base): VGPR pair via "v".
    //   - `byte_offset_imm` (compile-time INST.OFFSET literal): "n".
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    const uint32_t lds_byte_offset = (uint32_t)((uintptr_t)lptr);
#pragma clang diagnostic pop
    // Wave-uniform readfirstlane keeps `m0` an SGPR even if optimizer didn't
    // see the wave-uniformity (our caller does pass a wave-uniform value).
    const uint32_t lds_byte_offset_u =
        __builtin_amdgcn_readfirstlane(lds_byte_offset);
    uint32_t m0_dep;
    asm volatile("s_mov_b32 m0, %1"
                 : "=s"(m0_dep) // SSA tie-back into the load asm's input
                 : "s"(lds_byte_offset_u)
                 : "memory");

    if constexpr(bytes == 16)
        asm volatile("global_load_lds_dwordx4 %0, off offset:%c1"
                     :
                     : "v"(gptr), "n"(byte_offset_imm), "s"(m0_dep)
                     : "memory");
    else if constexpr(bytes == 12)
        asm volatile("global_load_lds_dwordx3 %0, off offset:%c1"
                     :
                     : "v"(gptr), "n"(byte_offset_imm), "s"(m0_dep)
                     : "memory");
    else /* bytes == 4 */
        asm volatile("global_load_lds_dword %0, off offset:%c1"
                     :
                     : "v"(gptr), "n"(byte_offset_imm), "s"(m0_dep)
                     : "memory");
#endif
}

} // namespace ck_tile
