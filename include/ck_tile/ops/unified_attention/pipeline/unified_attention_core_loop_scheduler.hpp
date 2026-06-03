// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// UA-owned counterpart of `CoreLoopScheduler` in
// `ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp`.
//
// The FMHA core-loop scheduler emits `__builtin_amdgcn_sched_group_barrier`
// hints that prescribe the per-phase mix of instruction types the compiler
// should pack into each `Scheduler::schedule(cl_p, number<Phase>{})` slot
// of the unified-attention pingpong loop. The hints assume the *baseline*
// placement of `fmha_mask` inside the K-side memory phase (W0-3 phase 1,
// W4-7 phase 2). If we move the mask onto the compute phase
// (gated by `MOVE_FMHA_MASK_TO_COMPUTE` in
// `unified_attention_pipeline.hpp`) without also moving the matching
// "2 VALU + 4 SALU" hint, the compiler oversubscribes the compute phase
// and undersubscribes the K-mem phase — measured on bf16 prefill_d128
// as a ~0.7% wall-clock regression and on FP8 prefill_d128 as a hard
// correctness failure (the FP8-only `cvt_pk_fp8 + ds_bpermute` cluster
// inside `fmha_alu1` is timing-sensitive and gets disrupted).
//
// To decouple UA from the FMHA scheduler, we fork the table here and
// thread the `MOVE_FMHA_MASK_TO_COMPUTE` switch through it: when the
// macro is 1, the "2 VALU + 4 SALU" entry is *moved* from the K-mem
// phase onto the compute phase (W0-3: 1 → 0; W4-7: 2 → 1) so the hint
// stays in lockstep with the code motion. When 0, the table is
// byte-identical to the FMHA one — same hints, same codegen,
// same perf as the upstream scheduler.

#pragma once

#include "ck_tile/core.hpp"

#ifndef MOVE_FMHA_MASK_TO_COMPUTE
#define MOVE_FMHA_MASK_TO_COMPUTE 0
#endif

#ifndef MOVE_FMHA_MASK_TO_GEMM1
#define MOVE_FMHA_MASK_TO_GEMM1 0
#endif

#ifndef CK_TILE_DISABLE_PACKED_FP32
#define CK_TILE_DISABLE_PACKED_FP32 0
#endif

#ifndef CONDITIONAL_RESCALE
#define CONDITIONAL_RESCALE 0
#endif

// Per-MFMA VALU hint for the gemm1 + fmha_alu_D_upd phase (W0-3 phase 2 /
// W4-7 phase 3). The baseline reserves 4 VALU slots per PV-MFMA to interleave
// the always-on 128-VGPR o_acc rescale tail. With CONDITIONAL_RESCALE the
// rescale is skipped on ~85% of KV tiles (wave-uniform branch), so most of
// those slots sit empty and the static hint over-reserves — leaving schedule
// bubbles the MFMAs could otherwise fill. Drop the per-MFMA VALU reservation
// (the score-shift VALU that is *always* present is small) so the compiler
// packs the common skip path tighter. Tuned empirically; revert to 4 if a
// shape regresses.
#if CONDITIONAL_RESCALE
#define UA_DUPD_PER_MFMA_VALU 2
#else
#define UA_DUPD_PER_MFMA_VALU 4
#endif

namespace ck_tile {

template <typename PipelineProblem, bool kIsMasking>
struct UAcoreLoopScheduler;

template <typename PipelineProblem>
struct UAcoreLoopScheduler<PipelineProblem, /*kIsMasking=*/true>
{
    template <ck_tile::index_t WaveGroup, ck_tile::index_t Phase>
    CK_TILE_DEVICE static constexpr void schedule(ck_tile::number<WaveGroup>,
                                                  ck_tile::number<Phase>)
    {
        using namespace ck_tile;

        if constexpr(WaveGroup == 0)
        {
            if constexpr(Phase == 0)
            {
                // gemm0 + fmha_alu1 [+ fmha_mask if MOVE_FMHA_MASK_TO_COMPUTE]
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // TRANS
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
#if MOVE_FMHA_MASK_TO_COMPUTE
                // Hint shifted here from the original phase-1 slot to match
                // the `fmha_mask` move out of the K-mem phase.
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
#endif
            }
            else if constexpr(Phase == 1)
            {
                // K_mem_load + V_lds_load [+ fmha_mask if baseline]
#if !MOVE_FMHA_MASK_TO_COMPUTE && !MOVE_FMHA_MASK_TO_GEMM1
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
#endif
            }
            else if constexpr(Phase == 2)
            {
                // [fmha_mask if MOVE_FMHA_MASK_TO_GEMM1] + gemm1 + fmha_alu_D_upd
#if MOVE_FMHA_MASK_TO_GEMM1
                // Hint shifted here from the original phase-1 slot to match
                // the `fmha_mask` move onto the head of the gemm1 phase
                // (just before cl_calc(p23, gemm1) which ends in
                //  fmha_alu0(p01_idx) and would otherwise read un-masked
                //  sp[p01].sp_compute).
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
#endif
#if !CK_TILE_DISABLE_PACKED_FP32
                __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // VALU
#endif
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                    // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x002, UA_DUPD_PER_MFMA_VALU, 0); // VALU
                });
            }
            else if constexpr(Phase == 3)
            {
                // V_mem_load + K_lds_load
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
        }
        else
        {
            if constexpr(Phase == 0)
            {
                // V_mem_load + K_lds_load
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
            else if constexpr(Phase == 1)
            {
                // gemm0 + fmha_alu1 [+ fmha_mask if MOVE_FMHA_MASK_TO_COMPUTE]
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // TRANS
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
#if MOVE_FMHA_MASK_TO_COMPUTE
                // Hint shifted here from the original phase-2 slot to match
                // the `fmha_mask` move out of the K-mem phase.
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
#endif
            }
            else if constexpr(Phase == 2)
            {
                // K_mem_load + V_lds_load [+ fmha_mask if baseline]
#if !MOVE_FMHA_MASK_TO_COMPUTE && !MOVE_FMHA_MASK_TO_GEMM1
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
#endif
            }
            else if constexpr(Phase == 3)
            {
                // [fmha_mask if MOVE_FMHA_MASK_TO_GEMM1] + gemm1 + fmha_alu_D_upd
#if MOVE_FMHA_MASK_TO_GEMM1
                // Hint shifted here from the original phase-2 slot to match
                // the `fmha_mask` move onto the head of the gemm1 phase
                // (just before cl_calc(p23, gemm1) which ends in
                //  fmha_alu0(p01_idx) and would otherwise read un-masked
                //  sp[p01].sp_compute).
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
#endif
#if !CK_TILE_DISABLE_PACKED_FP32
                __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // VALU
#endif
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                    // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x002, UA_DUPD_PER_MFMA_VALU, 0); // VALU
                });
            }
        }
    }
};

template <typename PipelineProblem>
struct UAcoreLoopScheduler<PipelineProblem, /*kIsMasking=*/false>
{
    // No-mask specialization is unaffected by MOVE_FMHA_MASK_TO_COMPUTE
    // (there's no mask to move), so this is byte-identical to the
    // FMHA `kIsMasking=false` table.
    template <ck_tile::index_t WaveGroup, ck_tile::index_t Phase>
    CK_TILE_DEVICE static constexpr void schedule(ck_tile::number<WaveGroup>,
                                                  ck_tile::number<Phase>)
    {
        using namespace ck_tile;

        if constexpr(WaveGroup == 0)
        {
            if constexpr(Phase == 0)
            {
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // TRANS
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
            }
            else if constexpr(Phase == 1)
            {
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
            else if constexpr(Phase == 2)
            {
#if !CK_TILE_DISABLE_PACKED_FP32
                __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // VALU
#endif
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                    // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x002, UA_DUPD_PER_MFMA_VALU, 0); // VALU
                });
            }
            else if constexpr(Phase == 3)
            {
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
        }
        else
        {
            if constexpr(Phase == 0)
            {
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
            else if constexpr(Phase == 1)
            {
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // TRANS
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
            }
            else if constexpr(Phase == 2)
            {
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
            else if constexpr(Phase == 3)
            {
#if !CK_TILE_DISABLE_PACKED_FP32
                __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // VALU
#endif
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                    // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x002, UA_DUPD_PER_MFMA_VALU, 0); // VALU
                });
            }
        }
    }
};

} // namespace ck_tile
