// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include "ck_tile/core.hpp"

// ─── ISA label markers for two-pass instruction-prefetch offset patching ───
// Used by script/patch_prefetch_offset.py to locate prefetch sites and targets
// in compiled GPU assembly and patch the koffset field.

// Stringify helpers (shared by INST_PREFETCH_TARGET and INST_PREFETCH)
// CK_TILE_STR_ stringifies directly; CK_TILE_XSTR_ expands macros first.
#ifndef CK_TILE_STR_
#define CK_TILE_STR_(x) #x
#define CK_TILE_XSTR_(x) CK_TILE_STR_(x)
#endif

// INST_PREFETCH_TARGET(label)       — default mode (mode=0): target is first instruction after
// comment. INST_PREFETCH_TARGET(label, mode) — mode=1 (BLOCK_ENTRY): script scans backward to
// nearest block
//                                    label (.LBB*:) and uses the first instruction after that.
//                                    Use when the compiler hoists ALU between the block entry
//                                    and the asm comment.
#define CK_PLACE_MODE_DEFAULT 0
#define CK_PLACE_MODE_BLOCK_ENTRY 1

#ifndef INST_PREFETCH_TARGET
#define INST_PREFETCH_TARGET_1(lbl) asm volatile("; [ck_label] name=" CK_TILE_STR_(lbl) " mode=0")
#define INST_PREFETCH_TARGET_2(lbl, mode) \
    asm volatile("; [ck_label] name=" CK_TILE_STR_(lbl) " mode=" CK_TILE_STR_(mode))

#define INST_PREFETCH_TARGET_GET_MACRO(_1, _2, NAME, ...) NAME
#define INST_PREFETCH_TARGET(...)                                                               \
    INST_PREFETCH_TARGET_GET_MACRO(__VA_ARGS__, INST_PREFETCH_TARGET_2, INST_PREFETCH_TARGET_1) \
    (__VA_ARGS__)
#endif

// INST_PREFETCH(label, num_cachelines)
// INST_PREFETCH(label, num_cachelines, direction)
// INST_PREFETCH(label, num_cachelines, direction, offset_cachelines)
// Emit the [ck_prefetch] comment AND s_prefetch_inst_pc_rel with koffset=0.
// num_cachelines: number of 128B cache lines to prefetch (klength = num_cachelines - 1).
// direction: CK_PREFETCH_DIR_FORWARD (default) or CK_PREFETCH_DIR_BACKWARD.
//   Forward:  INST_PREFETCH_TARGET marks the first cacheline of the prefetch region.
//   Backward: INST_PREFETCH_TARGET marks the last cacheline; the prefetch region extends
//             backward by num_cachelines from INST_PREFETCH_TARGET.
// offset_cachelines: additional offset in cachelines added to the computed koffset.
//   Allows multiple INST_PREFETCHes to share the same INST_PREFETCH_TARGET label but cover
//   different sub-regions, e.g. INST_PREFETCH(lbl, 32, DIR_FORWARD, 0) and
//   INST_PREFETCH(lbl, 32, DIR_FORWARD, 32) cover 64 cachelines total.
// The koffset is patched by script/patch_prefetch_offset.py in a second pass.
// Only emits code on gfx12+; on other targets it is a no-op.
#define CK_PREFETCH_DIR_FORWARD forward
#define CK_PREFETCH_DIR_BACKWARD backward

#ifndef INST_PREFETCH
#if defined(__gfx12__)
#define INST_PREFETCH_4(lbl, num_cachelines, direction, offset_cachelines)                       \
    do                                                                                           \
    {                                                                                            \
        asm volatile(                                                                            \
            "; [ck_prefetch] name=" CK_TILE_STR_(lbl) " dir=" CK_TILE_XSTR_(                     \
                direction) " offset=" CK_TILE_XSTR_(offset_cachelines) "\n\t"                    \
                                                                       "s_prefetch_inst_pc_rel " \
                                                                       "0, null, %0"             \
            :                                                                                    \
            : "n"((num_cachelines) - 1));                                                        \
    } while(false)
#define INST_PREFETCH_3(lbl, num_cachelines, direction) \
    INST_PREFETCH_4(lbl, num_cachelines, direction, 0)
#define INST_PREFETCH_2(lbl, num_cachelines) \
    INST_PREFETCH_3(lbl, num_cachelines, CK_PREFETCH_DIR_FORWARD)
#define INST_PREFETCH_GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#define INST_PREFETCH(...)                                                                  \
    INST_PREFETCH_GET_MACRO(__VA_ARGS__, INST_PREFETCH_4, INST_PREFETCH_3, INST_PREFETCH_2) \
    (__VA_ARGS__)
#else
#define INST_PREFETCH(lbl, ...)
#endif
#endif

// Enable scalar prefetch in hardware (required on gfx12 before using s_prefetch)
__device__ __forceinline__ void enable_scalar_prefetch()
{
#if defined(__gfx12__)
    // SCALAR_PREFETCH_EN is bit 24 in MODE register (hwreg 1)
    // Set 1 bit at offset 24 to value 1
    __builtin_amdgcn_s_setreg(1 | (24 << 6), 1);
#endif
}
