// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// EXPERIMENT: move the FA4 V LDS transpose-read out of the SOFTMAX phase and
// into the MATRIX phase (right before its PV consumer). The default ("Stage B")
// issues the V ds_read in the *preceding* softmax to hide its latency under the
// softmax VALU — but the ATT trace shows that read stalls ~69% and sits on the
// (longer/critical) softmax phase. Moving it to MATRIX, which has barrier slack,
// takes it off the critical path. =1 to enable.
#ifndef UA_FA4_PIN_PACK_IN_SOFTMAX
// Experiment (option 3): fence the fp32->fp8 P-pack (cvt_pk_fp8 tail of
// fmha_alu1) so it retires inside the SOFTMAX phase instead of sinking past the
// matrix barrier into the next MATRIX slot. In the MATRIX slot the pack (a VALU
// op) contends with the co-resident warp group's softmax VALU (the v_max3
// rowmax tree) on the shared SIMD issue port; pinning it to SOFTMAX trades that
// cross-wave contention for in-phase exposure on the (longer) softmax phase.
#define UA_FA4_PIN_PACK_IN_SOFTMAX 0
#endif

// UA_FA4_PREFETCH_IN_SOFTMAX: issue the next-tile K/V async DRAM prefetch from
// the SOFTMAX phase instead of the MATRIX phase. Gated to the 2-byte (bf16/fp16)
// path in the loop (fp8 keeps the matrix-phase prefetch it was tuned for). The
// bf16 prefetch is double the VMEM bytes and its buffer_load-to-LDS issue was
// landing in the (lgkm-stalled) MATRIX phase; moving the *issue* to the VALU-
// bound SOFTMAX phase keeps MATRIX pure-matrix. Residency is unchanged: the next
// MATRIX still drains the load via s_waitcnt_vmcnt<0> + block barrier before any
// K/V LDS read, so this only moves WHERE the async load is kicked off.
#ifndef UA_FA4_PREFETCH_IN_SOFTMAX
#define UA_FA4_PREFETCH_IN_SOFTMAX 1
#endif

// FMHA_MASK PLACEMENT: pick exactly one of:
//   - both 0 → baseline (mask in K-side memory phase, W0-3 phase 1
//     / W4-7 phase 2, right after `cl_load(memK)`).
//   - MOVE_FMHA_MASK_TO_COMPUTE=1: hoist mask onto the compute phase
//     (W0-3 phase 0 / W4-7 phase 1), right after `fmha_alu1`.
//     Experiment 1.5 finding: bf16 −0.33%, **fp8 +8.8% regression**
//     because the FP8 cvt+bperm cluster inside `fmha_alu1` makes the
//     compute phase already-saturated; adding T_mask oversubscribes
//     it and the empirical cost is ~2× the bare instruction count.
//   - MOVE_FMHA_MASK_TO_GEMM1=1: place mask at the START of the
//     gemm1 phase (W0-3 phase 2 / W4-7 phase 3), right before
//     `cl_calc(xdl_SP_p23_reg_idx, gemm1)`. This is the latest legal
//     placement: `cl_calc(p23, gemm1)` ends with `fmha_alu0(p01_idx)`
//     which READS `sp[p01_idx].sp_compute` to compute `m_latest`, so
//     mask MUST run before that. Phase 3 (V-mem on W0-3, gemm1 on
//     W4-7) is too late and silently corrupts the row-max.
//
//     For W4-7 the `++i_total_loops` also defers from end of phase 2
//     to start of phase 3 (after mask, before cl_calc) so mask sees
//     the same i_total_loops value as gemm0 of this iter.
//
//     Per-barrier algebra (mask added to gemm1 phase = T_D on both
//     warp groups, removed from K-mem = T_K on both):
//       - B1 wait = |T_C − (T_D + T_mask)|. With baseline T_C > T_D
//         on FP8, the gap closes — DECREASES by T_mask.
//       - B2 wait = |(T_K − T_mask) − T_C| — DECREASES by T_mask.
//       - B3 wait = |(T_D + T_mask) − (T_K − T_mask)|
//                 = |T_D − T_K + 2·T_mask| — DECREASES by 2·T_mask.
//       - Net: −4·T_mask total wait (vs −2·T_mask for compute), and
//         gemm1 phase has no FP8 cvt+bperm so it should absorb the
//         mask without the FP8 oversubscription that hit compute.
//
// Must be defined BEFORE including unified_attention_core_loop_scheduler.hpp
// — that header's `__builtin_amdgcn_sched_group_barrier` per-phase
// hints are gated on these macros and need to stay in lockstep with
// the code motion in this file.
#define MOVE_FMHA_MASK_TO_COMPUTE 0
#define MOVE_FMHA_MASK_TO_GEMM1   0
#if MOVE_FMHA_MASK_TO_COMPUTE && MOVE_FMHA_MASK_TO_GEMM1
#error "MOVE_FMHA_MASK_TO_COMPUTE and MOVE_FMHA_MASK_TO_GEMM1 are mutually exclusive"
#endif

// UA_DYNAMIC_SETPRIO (warp-group-balance plan A2, HipKittens-style)
//   0 (default): static per-warp-group priority, set once at loop entry
//     (W0-3 → s_setprio(0), W4-7 → s_setprio(1)). Baseline, bit-identical.
//   1: dynamic priority around the gemm MFMA cluster. `cl_calc` raises
//     s_setprio(1) for the duration of the gemm (QK/PV MFMAs + trailing
//     fmha_alu0) and drops back to s_setprio(0) after. The two warp groups
//     are offset by two phases and co-resident (one wave of each group per
//     SIMD), so the group currently in the compute cluster outbids the
//     group currently issuing memory for the shared VALU/MFMA issue port —
//     targeting the ARBITER_NOT_WIN stall that gates the compute side
//     (W0-3: 37.8% of its stalls). Under the macro the static W4-7=1 entry
//     is neutralised to 0 so the non-compute baseline is uniformly prio 0.
#ifndef UA_DYNAMIC_SETPRIO
#define UA_DYNAMIC_SETPRIO 0
#endif

// UA_FA4_PACKED_SHIFT: emit the softmax score-shift (sp_delta = sp_compute *
// scale_s - scale_s * rowmax) as packed v_pk_fma_f32 (2 f32/instr) instead of 64
// scalar v_fma_f32. Bit-identical: each thread holds one rowmax
// (m.thread_buf_.size()==1) so the FMA addend is uniform across the thread's
// score elements and is broadcast into both packed lanes. Mirrors the hand-tuned
// ASM softmax (v_pk_fma_f32 for the rebase). Halves the shift instruction count.
//
// MEASURED REGRESSION, default OFF. Together with UA_FA4_PACKED_ALU1_RESCALE this
// costs ~3% on the canonical fp8 prefill shape (GPU2, same-session 3-run median:
// packed 1825 TF/s vs scalar 1877 TF/s). The softmax score-shift is hidden under
// the ping-pong overlap, so collapsing the FMAs does not shorten the critical
// path; it only perturbs the scheduler and loses. (An earlier "+4.5%" reading was
// a confounded GPU0 measurement.) Kept gated off for documentation; do not enable.
#ifndef UA_FA4_PACKED_SHIFT
#define UA_FA4_PACKED_SHIFT 0
#endif

// UA_FA4_EXP2_APPROX: replace the per-element softmax exp (quarter-rate
// v_exp_f32) with the Schraudolph 2^x bit-trick (full-rate). The score-shift FMA
// in fmha_alu0 absorbs the 2^23 scale and the Schraudolph bias, so fmha_alu1 only
// needs a single v_cvt_u32_f32 per element instead of v_exp_f32. The per-row
// max-delta rescale keeps the exact v_exp_f32 (only 1/row, not on the hot path).
// This is an APPROXIMATION (~1e-3 rel error per element) -- it mirrors the ASM
// softmax fast SKU and is only applied on the non-masked, no-softcap path
// (compile-time gate below). Numerics-changing => default OFF; validate accuracy
// before enabling.
#ifndef UA_FA4_EXP2_APPROX
#define UA_FA4_EXP2_APPROX 0
#endif

// Schraudolph 2^x bit-trick constants: bits = round(2^23 * x + (127*2^23 - 486411))
// reinterpreted as f32 ~= 2^x. Min-error offset matches the hand-tuned ASM softmax.
#define UA_EXP2_SCHRAUDOLPH_SCALE 8388608.0f    // 2^23
#define UA_EXP2_SCHRAUDOLPH_BIAS 1064866805.0f  // 127*2^23 - 486411

// UA_FA4_PACKED_ROWSUM: reduce a thread's row-sum of probabilities with packed
// v_pk_add_f32 into a 2-wide partial, then a single scalar combine, instead of the
// scalar v_add_f32 dependency chain that block_tile_reduce emits. Halves the
// in-thread adds and shortens the latency chain feeding the cross-lane permlane.
// Reassociates the sum (rounding differs at the ULP level) -- safe within the fp8
// /bf16 attention tolerances.
//
// MEASURED LOSER (-13%): a serial v_pk_add_f32 accumulation is a 32-deep latency
// chain that is WORSE than block_tile_reduce's log-depth tree, and the dead scalar
// reduce is not DCE'd. Kept gated off for documentation; do not enable.
#ifndef UA_FA4_PACKED_ROWSUM
#define UA_FA4_PACKED_ROWSUM 0
#endif

// UA_FA4_PACKED_ALU1_RESCALE: pack the 6-register o_acc partial rescale in
// fmha_alu1 (elementwise *= o_acc_scale) with v_pk_mul_f32, matching the packed
// rescale in fmha_alu_D_upd. Independent elementwise scale (no dependency chain),
// and it halves the number of asm-volatile scheduling boundaries (6 scalar
// v_mul_f32 -> 3 v_pk_mul_f32). Bit-identical.
//
// MEASURED REGRESSION, default OFF -- see the note on UA_FA4_PACKED_SHIFT above:
// the two together cost ~3% (1877 -> 1825 TF/s, GPU2) because the rescale is
// hidden under the ping-pong overlap. (An earlier "+4%" reading was a confounded
// GPU0 measurement.) Kept gated off for documentation; do not enable.
#ifndef UA_FA4_PACKED_ALU1_RESCALE
#define UA_FA4_PACKED_ALU1_RESCALE 0
#endif

// CONDITIONAL_RESCALE (PLAN_conditional_rescale Part 2)
//   0 (default): always-rescale online softmax — the o_acc/l accumulators are
//     renormalised to the true running max `m` every KV tile (the expensive
//     128-VGPR `v_pk_mul_f32` rescale tail in fmha_alu_D_upd + the 6-reg
//     partial in fmha_alu1). Bit-identical to the pre-Part-2 kernel.
//   1: FA4-style conditional (skipped) rescale. Carry the accumulators in the
//     frame of a *committed* max `m_commit` that only advances (with a rescale)
//     when the true max pulls ahead by more than τ = log2 of the safe exp2
//     bound. Between commits the shifted scores stay ≤ τ (exp2 ≤ 2^τ, fp32-
//     safe) so o_acc/l just accumulate — the rescale multiplies are skipped.
//     The decision is made wave-uniformly (ballot: rescale if ANY lane needs
//     it) so the guard is a scalar branch with no divergence. Mathematically
//     exact (the m_commit frame cancels in o_acc/l; LSE uses m_commit), so no
//     end-of-loop correction is needed. Only applied on the 2-warp-group
//     prefill path (see kCondRescale); decode keeps always-rescale. Part-1's
//     --headroom instrument predicts ~85% (prefill) of rescales are skippable.
// Defined BEFORE the includes so unified_attention_core_loop_scheduler.hpp can
// gate its per-phase sched_group_barrier VALU hints on it (the gemm1+D_upd
// phase reserves ~36 VALU slots for the rescale tail that this skips).
#if !defined(CONDITIONAL_RESCALE)
#define CONDITIONAL_RESCALE 1
#endif
// τ in scaled-logit (log2) units. exp2(τ) bounds the un-rescaled scores; 8 =>
// max intermediate exp2 == 256, comfortably inside fp32 range even summed over
// thousands of keys. FA4 uses the same log2(256)=8.
#if !defined(CONDITIONAL_RESCALE_TAU)
#define CONDITIONAL_RESCALE_TAU 8.0f
#endif

#include "ck_tile/core.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_core_loop_scheduler.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#define ENABLE_ASM_MARKER 1
#if ENABLE_ASM_MARKER
#define ASM_MARKER(marker)               \
    __builtin_amdgcn_sched_barrier(0);   \
    asm volatile("; [POYENC] " #marker); \
    __builtin_amdgcn_sched_barrier(0);
#else
#define ASM_MARKER(marker)
#endif

// (ADD_SBARRIER_FOR_PHASE0 / ADD_SBARRIER_FOR_PHASE2 removed with the legacy
// ping-pong core_loop — they only gated barriers inside that deleted path.
// The FA4 pipeline manages its per-phase block barriers inline.)

// FA4 pipeline (FlashAttention-4 style matrix‖softmax warp-group overlap).
//
// FA4 is the ONLY 2-warp-group prefill pipeline. (The legacy 8-wave
// compute/memory "ping-pong" baseline — two groups offset by two phases, each
// running the monolithic interleaved core_loop — was REMOVED in the -fav4
// cleanup. It was selected only under -DUA_FA4_PIPELINE=0, which never beat
// FA4 on any measured prefill shape, so the toggle and the dead code path are
// gone.)
//
// Both warp groups run the *same* deferred-PV software pipeline (the
// known-correct sequence from the single-warp-group path: fmha_alu1(prev) →
// PV(prev) → QK(cur) → alu0(cur) → D_upd), cut into two phases —
//   MATRIX  phase: PV(k-1) + QK(k)                       (matrix pipe only)
//   SOFTMAX phase: alu1/exp + alu0/rowmax + D_upd/rescale (VALU/MUFU only)
// — and the two groups are primed one phase apart (WG0 enters MATRIX while WG1
// enters SOFTMAX) so on each SIMD the matrix-pipe work of one wave hides the
// VALU/transcendental work of its co-resident partner. The O*corr rescale
// (fmha_alu_D_upd) stays at the END of the SOFTMAX phase so the MATRIX phase is
// VALU-free (no MFMA-waits-on-rescale hazard, no cross-warp VALU contention).
// K/V are prefetched a tile ahead into a shared double buffer at the per-phase
// block barrier (issued cooperatively by all 8 warps so the full tile loads
// exactly once).
//
// Engages for the 2-warp-group prefill path only (NumWarpGroups == 2; bf16 /
// fp16, and FP8 on the 32x32x16 tiers where the P relayout is the barrier-free
// within-wave permute — see kFA4). Single-warp-group decode tiers keep their
// serial pipeline (kFA4 == false there). The kFA4 static_assert guarantees no
// 2-WG instance is left without a pipeline.

#if !defined(CK_TILE_DISABLE_PACKED_FP32)
#define CK_TILE_DISABLE_PACKED_FP32 0
#endif

#define WARP_ID 0
#define LANE_ID 0

#define ENABLE_DEBUG_STMTS 1
#if ENABLE_DEBUG_STMTS
#define DEBUG_STMTS \
    if(get_block_1d_id() == 0 && get_warp_id() == WARP_ID && get_lane_id() == LANE_ID)
#else
#define DEBUG_STMTS if constexpr(false)
#endif

namespace ck_tile {

// kPageSize_ : non-type template parameter that pins the runtime
// `page_size` argument to a compile-time constant when > 0. The host
// dispatcher selects an instance whose kPageSize_ matches `args.page_blk_size`
// and routes execution there; instances compiled with kPageSize_ == 0 keep
// the legacy runtime-page-size path and serve as the catch-all fallback for
// uncommon page sizes. Having the value at compile time:
//   1. lets the compiler strength-reduce every `/ page_size`, `* page_size`,
//      `% page_size` into shift / multiply-by-magic-constant on the literal
//      (e.g. div-by-32 → shr 5);
//   2. lets the Tier 0 / Tier 2 gate use the real `KY0_step_N <= kPageSize`
//      condition instead of the conservative `KY0_step_N <= 16` hedge, so
//      prefill_d128 bf16, prefill_d64 bf16, and prefill_d64 fp8 also gain
//      the scalar-promote + LDS-cache fast path on their natural page sizes.
template <typename Problem_,
          typename Policy_                  = UnifiedAttentionPipelineDefaultPolicy,
          ck_tile::index_t kPageSize_       = 0,
          bool kIsPaged_                    = true>
struct UnifiedAttentionPipeline
{
    using Problem             = ck_tile::remove_cvref_t<Problem_>;
    using Policy              = ck_tile::remove_cvref_t<Policy_>;

    // Compile-time page size (0 = runtime). See class-level comment above.
    static constexpr ck_tile::index_t kPageSize       = kPageSize_;
    static constexpr bool             kHasCePageSize = (kPageSize_ > 0);
    // Paged KV (block_tables indirection) vs contiguous/THD KV. When false the
    // K/V tile's logical token index IS its physical row (the per-sequence base
    // is folded into the K/V base pointer by the kernel), so all paging math —
    // block_tables fetch, page-table LDS cache, page-index arithmetic — is
    // compiled out and the load reduces to a plain linear scatter offset.
    static constexpr bool             kIsPaged       = kIsPaged_;
    using QDataType           = ck_tile::remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = ck_tile::remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = ck_tile::remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType        = ck_tile::remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType = ck_tile::remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using PDataType           = ck_tile::remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType        = ck_tile::remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType           = ck_tile::remove_cvref_t<typename Problem::ODataType>;
    using FmhaMask            = ck_tile::remove_cvref_t<typename Problem::FmhaMask>;

    static_assert(std::is_same_v<SaccDataType, SMPLComputeDataType>,
                  "we will the same dist tensor 'sp_compute' for both gemm0 & softmax");

    using UnifiedAttentionShape = ck_tile::remove_cvref_t<typename Problem::UnifiedAttentionShape>;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    static constexpr ck_tile::index_t kBlockM = UnifiedAttentionShape::kBlockM;
    static constexpr ck_tile::index_t kBlockQ = UnifiedAttentionShape::kBlockQ;

    static constexpr ck_tile::index_t kWarpGemmM =
        UnifiedAttentionShape::Gemm0WarpTile::at(ck_tile::number<0>{});

    static constexpr ck_tile::index_t kPageBlockSize = UnifiedAttentionShape::kPageBlockSize;
    static constexpr ck_tile::index_t kHeadDim       = UnifiedAttentionShape::kHeadDim;
    static constexpr ck_tile::index_t kHeadDimPadded = UnifiedAttentionShape::kHeadDimPadded;

    static_assert(kHeadDimPadded <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    // static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadHeadDimQ = Problem::kPadHeadDim;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDim;
    // static constexpr bool kStoreLSE    = Problem::kStoreLSE;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr ck_tile::index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    // The DRAM-view vector length must match the K/V load distribution's
    // KVector, which the FA4 decoupling widens to the per-warp-group load count
    // (GetK/VLoadNumWarps). Passing the same warp count here keeps the global
    // buffer_load width in lock-step with the async-copy descriptors.
    static constexpr ck_tile::index_t kAlignmentK =
        kPadHeadDimQ ? 1
                     : Policy::template GetAlignmentK<Problem,
                                                      Policy::template GetKLoadNumWarps<Problem>()>();
    static constexpr ck_tile::index_t kAlignmentV =
        kPadHeadDimV ? 1
                     : Policy::template GetAlignmentV<Problem,
                                                      Policy::template GetVLoadNumWarps<Problem>()>();

    static constexpr ck_tile::index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();

    static constexpr ck_tile::index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            return 2;
        }
    }();

    // Tier-2 LDS-resident page-table cache. Sized to cover sk up to
    // kPageTableLdsEntries * page_size tokens. 4096 entries × 4 B = 16 KiB,
    // which on gfx950 (160 KiB LDS/CU at kBlockPerCu=2 → 80 KiB/block)
    // is the available headroom on the prefill_d128 tier (~64 KiB existing
    // smem) without forcing kBlockPerCu down. Coverage envelope:
    //   page_size=16  →  sk ≤  65 536 tokens
    //   page_size=32  →  sk ≤ 131 072 tokens
    //   page_size=64  →  sk ≤ 262 144 tokens
    // Beyond that the kernel asserts on `num_pages_for_cta <=
    // kPageTableLdsEntries`. A runtime fallback was tried earlier and
    // regresses 30% because the compiler emits both refresh_*_offsets
    // paths and the resulting register pressure halves occupancy — so we
    // trap rather than silently miscompute. With the constexpr-page-size
    // (Tier 3) refactor this could be improved by scaling the cache size
    // by kPageSize (e.g. `64 KiB / kPageSize` entries) — TODO if a
    // page_size=16 long-context workload becomes important.
    static constexpr ck_tile::index_t kPageTableLdsEntries = 4096;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetPageTableLdsBytes()
    {
        // Allocate the cache only for the kernel instances where Tier 0's
        // constexpr gate fires (otherwise the lambdas wouldn't read it and
        // the LDS would sit idle, hurting occupancy for nothing). Mirror the
        // exact gate from operator() below — including the constexpr
        // page_size cap, which is the real precondition.
        using KDist = decltype(Policy::template MakeKDramTileDistribution<Problem>());
        using VDist = decltype(Policy::template MakeVDramTileDistribution<Problem>());
        constexpr ck_tile::index_t KNRepeat =
            Policy::kKNContigLoad
                ? KDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<2>{}]
                : KDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<0>{}];
        constexpr ck_tile::index_t VNRepeat =
            VDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<0>{}];
        constexpr ck_tile::index_t KY0_step_N =
            Policy::kKNContigLoad
                ? 1
                : KDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<1>{}] *
                      KDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<2>{}];
        constexpr ck_tile::index_t VY0_step_N =
            VDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<1>{}] *
            VDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<2>{}];
        constexpr ck_tile::index_t kPageSizeCap =
            kHasCePageSize ? kPageSize : ck_tile::index_t{16};
        // Gate kept in lock-step with kKNeedsPageTableLds / kScalarPromoteVPageIdx
        // in operator(): both decide whether a given kernel instance needs the
        // Tier-2 LDS-resident page-table cache, and any divergence means the
        // runtime path writes/reads at an offset for which no LDS was
        // reserved (silently corrupting the K/V double-buffers above it).
        //
        // Warp-major K reads its per-wave page index straight from global, so it
        // never needs the cache — must match kRebaseKSrdWarpMajor in operator().
        constexpr ck_tile::index_t kTokensPerWarpK =
            kPageBlockSize / Policy::template GetKLoadNumWarps<Problem>();
        constexpr bool kKRebaseWarpMajor = Policy::kKNContigLoad && kIsPaged &&
                                           kHasCePageSize && (kTokensPerWarpK <= kPageSize);
        // Mirror of kFallbackUsesLdsK / kMultiPageDedupK in operator(): the
        // multi-page K fallback opt-ins also consume the Tier-2 LDS cache.
        constexpr bool kScalarPromoteK =
            (KNRepeat >= 2) && (KY0_step_N <= kPageSizeCap);
        constexpr bool kFallbackLdsK = Policy::kKFallbackLds && kIsPaged &&
            kHasCePageSize && !kScalarPromoteK && !kKRebaseWarpMajor;
        constexpr bool kMultiPageDedupK = Policy::kKMultiPageDedup && kIsPaged &&
            kHasCePageSize && !kScalarPromoteK && !kKRebaseWarpMajor &&
            (KY0_step_N % kPageSize == 0);
        constexpr bool kHasTier0K =
            (kScalarPromoteK && !kKRebaseWarpMajor) || kFallbackLdsK || kMultiPageDedupK;
        constexpr bool kScalarPromoteV =
            (VNRepeat >= 2) && (VY0_step_N <= kPageSizeCap);
        constexpr bool kFallbackLdsV =
            Policy::kKFallbackLds && kIsPaged && kHasCePageSize && !kScalarPromoteV;
        constexpr bool kHasTier0V = kScalarPromoteV || kFallbackLdsV;
        if constexpr (kHasTier0K || kHasTier0V)
            return kPageTableLdsEntries * sizeof(ck_tile::index_t);
        else
            return 0;
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // Two layouts share the same smem base region during the pipeline:
        //   - the o_lds tile (kBlockM * kHeadDimPadded * sizeof(PDataType)),
        //     overlapped with the s_lds tile (kBlockM * kPageBlockSize *
        //     sizeof(SaccDataType)) — the QK gemm writes its float32 sacc
        //     into s_lds while the PV output is staged through o_lds;
        //   - K/V double-buffered storage (Policy::GetSmemSize) plus the
        //     p_lds tile that lives immediately after it (kBlockM *
        //     kPageBlockSize * sizeof(PDataType)).
        // For BF16/FP16 the o_lds term dominates so historically the
        // s_lds bound was implicit; for FP8 the PDataType drops to 1 B
        // while SaccDataType stays at 4 B, so the s_lds term becomes the
        // tightest bound and we must include it explicitly. The
        // static_assert further down (sizeof(SaccDataType) * kPageBlockSize
        // * kBlockM <= GetSmemSize()) now passes for every FP8 variant we
        // compile.
        // The Tier-2 page-table cache (if any) is appended at the very end
        // of the smem region so the existing layouts above are untouched.
        return ck_tile::max(ck_tile::max(kBlockM * kHeadDimPadded * sizeof(PDataType),
                                         kBlockM * kPageBlockSize * sizeof(SaccDataType)),
                            Policy::template GetSmemSize<Problem>() +
                                kBlockM * kPageBlockSize * sizeof(PDataType)) +
               GetPageTableLdsBytes();
    }

    // for debug only
    template <ck_tile::index_t MPerBlock, ck_tile::index_t NPerBlock>
    CK_TILE_DEVICE static constexpr auto MakeSimpleLdsDesc()
    {
        using namespace ck_tile;
        constexpr auto lds_block_desc =
            make_naive_tensor_descriptor(make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                                         make_tuple(number<NPerBlock>{}, number<1>{}),
                                         number<1>{},
                                         number<1>{});

        return lds_block_desc;
    }

    // for debug only
    template <ck_tile::index_t MPerBlock>
    CK_TILE_DEVICE static constexpr auto MakeSimpleLdsDesc1D()
    {
        using namespace ck_tile;
        constexpr auto lds_block_desc = make_naive_tensor_descriptor(
            make_tuple(number<MPerBlock>{}), make_tuple(number<1>{}), number<1>{}, number<1>{});

        return lds_block_desc;
    }

    template <typename DataType, typename Descriptor>
    CK_TILE_DEVICE static constexpr auto make_lds_tile_window(void* base, const Descriptor& desc)
    {
        using namespace ck_tile;

        auto tensor_view =
            make_tensor_view<address_space_enum::lds>(reinterpret_cast<DataType*>(base), desc);
        return make_tile_window(tensor_view, desc.get_lengths(), {0, 0});
    }

    // vmcnt=0~63, lgkmcnt=0~15, expcnt=0~7
    template <uint16_t Vmcnt, uint8_t Lgkmcnt, uint8_t Expcnt = 7>
    CK_TILE_DEVICE static constexpr void s_waitcnt()
    {
        // vmcnt use bits {[15:14],[3:0]}
        // expcnt use bits [6:4]
        // lgkmcnt use bits [11:8]
        __builtin_amdgcn_s_waitcnt((((0b110000 & Vmcnt) << (14 - 4)) | (0b1111 & Vmcnt)) |
                                   ((0b111 & Expcnt) << 4) | ((0b1111 & Lgkmcnt) << 8));
    }

    template <uint16_t Vmcnt>
    CK_TILE_DEVICE static constexpr void s_waitcnt_vmcnt()
    {
        s_waitcnt<Vmcnt, 15>();
    }

    template <uint8_t Lgkmcnt>
    CK_TILE_DEVICE static constexpr void s_waitcnt_lgkmcnt()
    {
        s_waitcnt<63, Lgkmcnt>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction>
    CK_TILE_DEVICE auto operator()(
        const QDramBlockWindowTmp& q_dram_block_window_tmp, // kBlockM * kHeadDimPadded tile
        const QElementFunction& q_element_func,
        const KDramBlockWindowTmp& k_dram_block_window_tmp, // kPageBlockSize * kHeadDimPadded tile
        [[maybe_unused]] const KElementFunction& k_element_func,
        const VDramBlockWindowTmp& v_dram_block_window_tmp, // kHeadDimPadded * kPageBlockSize tile
        [[maybe_unused]] const VElementFunction& v_element_func,
        const index_t num_blocks,
        const index_t num_blocks_start,
        const void* block_tables_ptr,
        index_t block_table_offset,
        // Runtime page size. Ignored when the pipeline was instantiated with
        // a non-zero kPageSize_ template arg (we assert below that the two
        // values match in that case); the body always reads through the
        // local `page_size` below, which is either kPageSize_ (constexpr,
        // folded into / * %) or this runtime value.
        const index_t page_size_runtime,
        [[maybe_unused]] const SAccElementFunction& s_acc_element_func,
        const PComputeElementFunction& p_compute_element_func,
        const OAccElementFunction& o_acc_element_func,
        FmhaMask mask,
        float scale_s,
        void* smem_ptr,
        long_index_t k_row_stride         = 0,
        long_index_t v_row_stride         = 0,
        // Runtime kBlockQ = kBlockM / num_queries_per_kv. Default of 0 means
        // "fall back to the compile-time `kBlockQ` from `UnifiedAttentionShape`"
        // so existing callers don't have to change. The kernel template passes
        // the runtime value (from kargs) to remove the static dependency.
        const index_t num_queries_per_kv = 0,
        // Caller-supplied flag: set to true when the K/V cache total byte
        // size can exceed INT32_MAX. Routes K/V async loads through the
        // 64-bit-base `global_load_lds` path (correct but lower throughput).
        // False uses the original shared-SRD `buffer_load_dword_lds` path.
        const bool cache_ptr_int32_overflow_possible = false,
        // Per-tensor FP8 V descale, applied to o_acc once after the 1/l
        // normalisation. For non-FP8 dtypes the host passes 1.0f and this
        // becomes a no-op multiply. The mathematical identity is:
        //   sum_j P[i,j] * V_real[j,:] = v_descale * sum_j P[i,j] * V_fp8[j,:]
        // so deferring the v_descale outside the K/V loop is exact (not an
        // approximation). For split-KV (num_splits > 1) each partial gets
        // v_descale baked in; the combine step's affine weighting passes it
        // through unchanged so the final O is correct.
        const float v_descale = 1.0f) const
    {
        using namespace ck_tile;
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        // Bind the rest of the body to a single `page_size` symbol. When
        // kPageSize_ > 0 the ternary collapses at compile time to the
        // template literal — every `x / page_size`, `x * page_size`, and
        // `x % page_size` below then gets strength-reduced (e.g. `/ 32` →
        // `shr 5`, `* 32` → `shl 5`). When kPageSize_ == 0 the ternary
        // collapses to the runtime value (legacy behaviour). A debug-build
        // assert catches the case where the host dispatcher picks the
        // wrong constexpr instance.
        if constexpr (kHasCePageSize) { assert(page_size_runtime == kPageSize); }
        const index_t page_size = kHasCePageSize ? kPageSize : page_size_runtime;

        static_assert(
            kBlockM == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                kPageBlockSize == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                kHeadDimPadded == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                kPageBlockSize == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                kHeadDimPadded == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
            "wrong!");

        static_assert(sizeof(SaccDataType) * kPageBlockSize * kBlockM <= GetSmemSize());
        auto s_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<SaccDataType*>(static_cast<char*>(smem_ptr)),
            MakeSimpleLdsDesc<kBlockM, kPageBlockSize>());
        [[maybe_unused]] auto s_lds_window = make_tile_window(
            s_lds, make_tuple(number<kBlockM>{}, number<kPageBlockSize>{}), {0, 0});

        auto p_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<PDataType*>(static_cast<char*>(smem_ptr) +
                                         Policy::template GetSmemSize<Problem>()),
            MakeSimpleLdsDesc<kBlockM, kPageBlockSize>());
        [[maybe_unused]] auto p_lds_window = make_tile_window(
            p_lds, make_tuple(number<kBlockM>{}, number<kPageBlockSize>{}), {0, 0});

        auto o_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<PDataType*>(static_cast<char*>(smem_ptr)),
            MakeSimpleLdsDesc<kBlockM, kHeadDimPadded>());
        [[maybe_unused]] auto o_lds_window = make_tile_window(
            o_lds, make_tuple(number<kBlockM>{}, number<kHeadDimPadded>{}), {0, 0});

        auto m_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<SMPLComputeDataType*>(static_cast<char*>(smem_ptr) +
                                                   Policy::template GetSmemSize<Problem>()),
            MakeSimpleLdsDesc1D<kBlockM>());
        [[maybe_unused]] auto m_lds_window =
            make_tile_window(m_lds, make_tuple(number<kBlockM>{}), {0});

        const index_t warp_group_id = get_warp_id() / 4;

        // FA4 "WG0 loads V": warp group 0's 4 waves load the FULL V tile into
        // the shared V LDS buffer (V descriptors use VLoadNumWarps == 4 waves);
        // warp group 1 skips the V DRAM load and relies on the inter-phase
        // barrier for residency. v_load_active gates the async V load issue.
        constexpr index_t VLoadNumWarps     = Policy::template GetVLoadNumWarps<Problem>();
        constexpr index_t KLoadNumWarps     = Policy::template GetKLoadNumWarps<Problem>();
        constexpr index_t NumWarpGroups_     = Problem::kBlockSize / Policy::NumThreadPerWarpGroup;
        const bool v_load_active =
            (!Policy::kFA4WG0LoadsV) || (NumWarpGroups_ != 2) || (warp_group_id == 0);
        // Symmetric: warp group 1 alone loads K (WG0 reads from shared LDS).
        const bool k_load_active =
            (!Policy::kFA4WG1LoadsK) || (NumWarpGroups_ != 2) || (warp_group_id == 1);

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        // -----------------------------------------------------------------
        // FP8 P-tile re-layout windows (LDS roundtrip).
        //
        // The QK-gemm C output and PV-gemm A input share the
        // `sp_compute` / `p` register union. For BF16 / FP16 the two
        // warp-gemms agree byte-for-byte on the per-thread element
        // ordering, so the union "just works". For FP8 the PV gemm is
        // forced into `WGAttrNumAccess::Single` (load_tile_transpose's
        // SubMinDim=8 constraint, see GetPVBlockGemm in the policy) and
        // the QK-C / PV-A per-thread layouts diverge — naively reusing
        // the union feeds garbled data to the PV gemm.
        //
        // The fix is layout-agnostic: after FP8 packing in fmha_alu1
        // we (1) view the union's bytes as a distributed tensor in the
        // QK-C distribution, (2) `store_tile` it through the existing
        // `p_lds` region in canonical (M, N) order, (3) block-sync, and
        // (4) `load_tile` back with the PV-A distribution into the
        // `sp(idx).p` register file. Both the 32x32x16 (prefill +
        // decode_m{32,64,128}) and the 16x16x32 (decode_m16) MFMAs
        // are handled uniformly.
        //
        // Two distribution-bound windows over the same `p_lds` view —
        // hoisted out of fmha_alu1 so we only pay the make_tile_window
        // cost once per kernel invocation:
        [[maybe_unused]] auto p_lds_store_window_qkc = make_tile_window(
            p_lds_window,
            decltype(gemm_0.MakeCBlockTile())::get_tile_distribution());
        [[maybe_unused]] auto p_lds_load_window_pva = make_tile_window(
            p_lds_window,
            Policy::template MakePRegTileDistribution<Problem>());

        auto q_dram_window = make_tile_window_linear(
            q_dram_block_window_tmp, Policy::template MakeQRegTileDistribution<Problem>());

        // auto q_dram_window = q_dram_block_window_tmp;
        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        constexpr index_t KStoreWarpShift = Policy::template GetKStoreWarpShift<Problem>();
        auto k_lds_window_store = generate_tuple(
            [&](auto i_buf) {
                return make_lds_tile_window<KDataType>(
                    smem_ptr,
                    Policy::template MakeKLdsStoreBlockDescriptor<Problem,
                                                                  KLoadNumWarps,
                                                                  KStoreWarpShift>(i_buf));
            },
            number<2>{});

        auto v_lds_window_store = generate_tuple(
            [&](auto i_buf) {
                return make_lds_tile_window<KDataType>(
                    smem_ptr,
                    Policy::template MakeVLdsStoreBlockDescriptor<Problem, VLoadNumWarps>(i_buf));
            },
            number<2>{});

        statically_indexed_array<
            decltype(make_tile_window(
                make_lds_tile_window<KDataType>(
                    nullptr,
                    Policy::template MakeKLdsLoadBlockDescriptor<Problem, KLoadNumWarps>()),
                Policy::template MakeKRegTileDistribution<Problem>())),
            2>
            k_lds_window_load;

        statically_indexed_array<
            decltype(make_tile_window(
                make_lds_tile_window<VDataType>(
                    nullptr,
                    Policy::template MakeVLdsLoadBlockDescriptor<Problem, VLoadNumWarps>()),
                Policy::template MakeVRegTileDistribution<Problem>())),
            2>
            v_lds_window_load;

        decltype(make_static_distributed_tensor<QDataType>(
            Policy::template MakeQRegTileDistribution<Problem>())) q_tile;

        // NOTE: k_tile / v_tile were historically a *union* to save VGPRs. But
        // occupancy here is LDS-bound (48KB/WG -> 1 WG/CU regardless), so the
        // union bought no occupancy yet forced a hard serialization: K_lds_load
        // writes the same VGPRs the PV MFMA reads (v_tile), so the K ds_read
        // could not start until the PV MFMA fully retired -> full LDS latency
        // exposed at the QK-gemm's s_waitcnt_lgkmcnt<0> (ATT: ~half of memwait).
        // Separate tiles let the K ds_read execute on the LSU *concurrently*
        // with the PV MFMA (it stays at the same program point, so the
        // cooperative-load residency slack is preserved -- see fa4_matrix).
        struct kv_tile_type
        {
            CK_TILE_DEVICE kv_tile_type() {}

            decltype(load_tile(k_lds_window_load(number<0>{}))) k_tile;

            decltype(load_tile_transpose(v_lds_window_load(number<0>{}))) v_tile;
        } kv_tile;

        union sp_compute_type
        {
            CK_TILE_DEVICE sp_compute_type() {}

            decltype(gemm_0.MakeCBlockTile()) sp_compute;
            decltype(make_static_distributed_tensor<PDataType>(
                Policy::template MakePRegTileDistribution<Problem>())) p;
        };
        // Collapse the deferred-PV score/P double buffer to a single shared slot
        // for the large (kv128) score tile. The 2-slot fp32 score/P tile is the
        // dominant kv128 VGPR consumer (~122 spills at kPageBlockSize=128);
        // single-buffering it fits under the 256-VGPR ceiling with 0 spills. This
        // is correctness-neutral: the deferred-PV PV(pi)-read and QK(1-pi)-write
        // now alias the same VGPRs -- a register WAR hazard the compiler resolves
        // by serializing PV->QK (pure-VGPR, no LDS/barrier). Small tiles
        // (kPageBlockSize<=64, decode) keep the double buffer: single-buffering
        // there would force that serialization for no spill benefit. The accessor
        // ignores the slot index so every sp(number<I>{}) call site compiles.
        // UA_FA4_SINGLE_SP forces it on regardless (probing).
#ifndef UA_FA4_SINGLE_SP
#define UA_FA4_SINGLE_SP 0
#endif
        static constexpr bool kUseSingleSp = (UA_FA4_SINGLE_SP != 0) || (kPageBlockSize >= 128);
        struct sp_holder_t
        {
            sp_compute_type s_;
            CK_TILE_DEVICE constexpr sp_compute_type& operator()(index_t) { return s_; }
        };
        std::conditional_t<kUseSingleSp,
                           sp_holder_t,
                           statically_indexed_array<sp_compute_type, 2>>
            sp;

        decltype(gemm_1.MakeCBlockTile()) o_acc;
        // threshold to decide how many fmha_alu_D_upd() o_acc-rescale registers are
        // moved into fmha_alu1(); overridable for split-ratio sweeps.
#ifndef UA_FA4_ALU_D_REG_CNT
#define UA_FA4_ALU_D_REG_CNT 6
#endif
        constexpr index_t fmha_alu_D_reg_cnt = UA_FA4_ALU_D_REG_CNT;
        static_assert(fmha_alu_D_reg_cnt <= o_acc.thread_buf_.size());

        decltype(block_tile_reduce<SMPLComputeDataType>(
            sp(number<0>{}).sp_compute, sequence<1>{}, f_max, SMPLComputeDataType{0})) m;
        decltype(m) l;
#if CONDITIONAL_RESCALE
        // Committed max the o_acc/l accumulators are normalised against, and
        // its value before the current tile's (possible) advance. `m_commit`
        // only moves when the wave decides to rescale (see fmha_alu0). Declared
        // here (alongside m/l) so the pre-loop init below can reach it.
        decltype(m) m_commit;
        decltype(m) m_commit_old;
        // Wave-uniform "this tile advanced m_commit" flag. Set in fmha_alu0,
        // consumed by fmha_alu_D_upd (o_acc[6:]) of the same tile and, via the
        // deferred carry, by the next fmha_alu1 (o_acc[0:6]). Uniform across
        // the wave so the rescale guard compiles to a scalar s_cbranch.
        bool need_rescale = true;
#endif

        // initialize k_lds_window and v_lds_window
        static_for<0, 2, 1>{}([&](auto idx) {
            k_lds_window_load(idx) = make_tile_window(
                make_lds_tile_window<KDataType>(
                    static_cast<char*>(smem_ptr) + (idx)*Policy::template GetSmemSizeKV<Problem>(),
                    Policy::template MakeKLdsLoadBlockDescriptor<Problem, KLoadNumWarps>()),
                Policy::template MakeKRegTileDistribution<Problem>());
        });

        static_for<0, 2, 1>{}([&](auto idx) {
            v_lds_window_load(idx) =
                make_tile_window(make_lds_tile_window<VDataType>(
                                     static_cast<char*>(smem_ptr) +
                                         (idx + 2) * Policy::template GetSmemSizeKV<Problem>(),
                                     Policy::template MakeVLdsLoadBlockDescriptor<Problem,
                                                                                 VLoadNumWarps>()),
                                 Policy::template MakeVRegTileDistribution<Problem>());
        });

        {
            auto origin_q      = load_tile(q_dram_window);
            auto transformed_q = tile_elementwise_in(q_element_func, origin_q);

            q_tile = transformed_q;
        }

        clear_tile(o_acc);
        set_tile(m, bit_cast<float>(0xff7fffff)); // a bit larger than -infinity
        clear_tile(l);
#if CONDITIONAL_RESCALE
        // Same -inf-ish init as `m`: the first tile's gap (m - m_commit) is
        // huge so it always commits, with m_commit_old == -inf giving
        // o_acc_scale == exp2(-inf) == 0 — a no-op on the cleared o_acc/l,
        // matching the always-rescale path's first-tile behaviour.
        set_tile(m_commit, bit_cast<float>(0xff7fffff));
        set_tile(m_commit_old, bit_cast<float>(0xff7fffff));
#endif

        const auto q_origin = q_dram_window.get_window_origin();

        const auto num_total_loop = num_blocks;
        index_t k_block_idx       = 0;
        index_t v_block_idx       = 0;

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking)
        {
            if(num_total_loop - num_blocks_start <= 0)
            {
                // Note: o_acc is already cleared above. q loaded but no fence
                // (ignored). lse must be -infinity so the split-KV combine
                // weighs this empty partial as zero (exp(-inf) == 0); for
                // single-split callers the value is harmless (ignored).
                auto lse_early =
                    make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
                set_tile(lse_early, -ck_tile::numeric<SMPLComputeDataType>::infinity());
                return ck_tile::make_tuple(o_acc, lse_early);
            }
        }

        index_t i_total_loops = num_blocks_start;
        const ck_tile::index_t* block_tables_ptr_ =
            reinterpret_cast<const ck_tile::index_t*>(block_tables_ptr);
        assert(k_block_idx == v_block_idx);
        // Split-KV start offset in *tokens* (not in tiles or pages). We add
        // this to logical_token below so the page-table lookup uses the right
        // page; we do NOT shift block_table_offset because num_blocks_start is
        // counted in kPageBlockSize-sized tiles, while block_tables is indexed
        // in page_size-sized pages — the two differ whenever kPageBlockSize !=
        // page_size and shifting tiles-as-pages reads the wrong entries.
        const index_t split_token_offset = num_blocks_start * kPageBlockSize;

        // Pass-2: unified page-offset formula. The kPageBlockSize <= page_size
        // constraint is gone. For every (thread, Y0-iter) pair we compute:
        //
        //     logical_token = tile_idx * kPageBlockSize
        //                   + thread_N_pos                 // lane/warp partition
        //                   + i * Y0_step_N                // per-Y0-iter advance
        //     logical_page  = logical_token / page_size    // index into block_tables
        //     within_page   = logical_token % page_size    // row inside the page
        //     phys_page     = block_tables[block_table_offset + logical_page]
        //     page_offsets[i] = (phys_page * page_size + within_page) * row_stride
        //
        // The page indirection moves entirely into page_offsets, so the per-iter
        // SRD rebase (set_bottom_tensor_view_data_ptr + init_raw) is dropped —
        // we just call update_page_idx() to refresh offsets between tiles. This
        // works for any (kPageBlockSize, page_size) pair where Y0_step_N (= the
        // inner N stride from the dist encoding, N1 * N2) divides page_size, so
        // a single wave-wide load instruction never straddles a page boundary.
        // If page_size < Y0_step_N, per-lane VGPR SRDs would be required and we
        // don't currently support that.
        //
        // TODO(overflow): page_offsets are index_t (int32). For caches whose
        // num_blocks * page_size * row_stride exceeds INT32_MAX, the offsets
        // wrap and reads return wrong data. The previous pass had a one-shot
        // base-pointer shift heuristic for this case (`use_ptr_rebase`); it has
        // been removed here because it does not interact well with the unified
        // formula when block_tables are non-monotonic (a far-away page produces
        // a large negative relative offset that the HW OOB check clamps to 0).
        // A robust fix would either plumb long_index_t through the gather load
        // path or compute a per-batch min-page shift in a pre-pass.
        const auto k_dist = Policy::template MakeKDramTileDistribution<Problem, KLoadNumWarps>();
        const auto v_dist = Policy::template MakeVDramTileDistribution<Problem, VLoadNumWarps>();
        using KDstrType   = decltype(k_dist);
        using VDstrType   = decltype(v_dist);
        // Warp-major K load (issue-fastest) reorders H0 to
        // <NumWarps, LaneGroups, NumIssues>, so the issue (Y0) dim is H0[2] (the
        // finest N factor) with per-issue token stride 1; KNRepeat is its extent.
        // The default layout keeps issue at H0[0] with stride H0[1]*H0[2]
        // (= LaneGroups*NumWarps).
        constexpr index_t KNRepeat =
            Policy::kKNContigLoad
                ? KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<2>{}]
                : KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<0>{}];
        constexpr index_t VNRepeat =
            VDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<0>{}];
        constexpr index_t KY0_step_N =
            Policy::kKNContigLoad
                ? 1
                : KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<1>{}] *
                      KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<2>{}];
        constexpr index_t VY0_step_N =
            VDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<1>{}] *
            VDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<2>{}];

        // WG-relative warp index for the gather page-offset computation. When a
        // single warp group loads a tile (V by WG0 / K by WG1), only that
        // group's waves issue the load, and their absolute warp ids must be
        // folded back into [0, NumWarps) so k_thread_n_pos / v_thread_n_pos (the
        // per-wave token position baked into page_idx) match the group-relative
        // distribution. For the cooperative case NumWarps == full block, so the
        // modulo is the identity. The scatter-gather's own get_partition_index
        // use is harmless here: the gather (token) dim is zeroed and replaced by
        // page_idx, and the remaining (head-dim) coordinate is lane-based.
        const auto k_part = ck_tile::array<index_t, 2>{get_warp_id() % KLoadNumWarps, get_lane_id()};
        const auto v_part = ck_tile::array<index_t, 2>{get_warp_id() % VLoadNumWarps, get_lane_id()};
        const auto k_thread_coord    = k_dist.calculate_index(k_part);
        const auto v_thread_coord    = v_dist.calculate_index(v_part);
        const index_t k_thread_n_pos = k_thread_coord[number<0>{}];
        const index_t v_thread_n_pos = v_thread_coord[number<0>{}];

        // Page offsets are widened to long_index_t so the `_long` load path
        // (global_load_lds, per-lane 64-bit base) can address pools whose
        // `num_blocks * page_size * row_stride * sizeof(T)` exceeds INT32_MAX.
        // Small-domain values (logical_token, logical_page, within_page,
        // phys_page) stay int32 — they're bounded by the per-CTA sequence
        // and never overflow. The original `async_load_tile_raw` path
        // implicitly narrows this back to int32 when it forwards the value
        // through `async_get_vectorized_elements_raw` — that's intentional,
        // and safe whenever `cache_ptr_int32_overflow_possible == false`.
        statically_indexed_array<long_index_t, KNRepeat> k_page_offsets;
        statically_indexed_array<long_index_t, VNRepeat> v_page_offsets;

        // Scalar-promote the block_tables[] lookup.
        //
        // The block_tables entry consumed by `refresh_*_offsets(tile_idx)` is
        // *uniform across the warp* for every `i`: the existing precondition
        // documented above ("Y0_step_N divides page_size") guarantees that the
        // per-lane spread k_thread_n_pos / v_thread_n_pos is strictly less
        // than page_size and contained within a single page. Without this
        // hint the compiler emits 64 redundant per-lane
        //   v_*_u32 (logical_page address math)
        //   global_load_dword
        //   s_waitcnt vmcnt(0)
        // per warp per K/V tile to materialise the same uint32, which on
        // long-context prefill dominates the kernel (~30-50% of every
        // LOAD_KV phase's samples; see ua-test-scripts/rocprof_analysis/
        // BOTTLENECK_ANALYSIS.md section 9).
        //
        // The fix: hoist the page-index arithmetic to compile-time-constant
        // strides + a single per-i SALU load (via __builtin_amdgcn_readfirstlane,
        // which both forces uniformity and lets LLVM lift the gather into
        // s_load_dword). The per-lane `within_page` calculation stays in VALU
        // because it has to.
        // Scalar-promote the block_tables[] page-index lookup.
        //
        // Idea: when the per-lane n-position span inside one distribution
        // issue (= KY0_step_N tokens) is contained in one page, the page
        // index is *uniform across the warp* for every `i`. Forcing the
        // page-table read through __builtin_amdgcn_readfirstlane gives LLVM
        // enough information to emit a single per-warp `s_load_dword` (with
        // a SALU address-comp chain) instead of 64 redundant per-lane
        //   v_*_u32 ... global_load_dword ... s_waitcnt vmcnt(0)
        // per K/V tile to materialise the same uint32 — which the bottleneck
        // analysis (see ua-test-scripts/rocprof_analysis/BOTTLENECK_ANALYSIS.md
        // section 9) flagged as ~30-50% of every LOAD_KV phase on long-context
        // prefill.
        //
        // Gating (compile-time only — a runtime fallback emits both paths into
        // the kernel and regresses ~30% from the resulting register pressure):
        //  (a) 8-warp blocks only: the 1-4 warp decode tiers regress 3-8%
        //      because the readfirstlane SALU op per `i` outweighs the win
        //      from collapsing per-lane VMEM loads that already coalesce
        //      well when only 1-4 warps participate.
        //  (b) KY0_step_N == kPageBlockSize/N for N>=2 ("at least 2 issues")
        //      AND KY0_step_N <= 16: needs the per-i n-span to fit in any
        //      sensible runtime page_size. Production vLLM/SGLang setups use
        //      page_size ∈ {16, 32, 64, 128}; 16 is the smallest commonly
        //      configured value so we cap at that. With the dwordx4 alignment
        //      fix this leaves prefill_d128 FP8 (KY0_step_N=16) as the lone
        //      qualifying tier; the other prefill variants (KY0_step_N ∈
        //      {32, 64}) stay on the per-lane path because in the absence of
        //      a host-side dispatch on page_size we cannot prove the span
        //      uniformity precondition at compile time.
        //
        // Measured win (sq=75600 prefill on MI355, n=80 iters x 3 trials):
        //   prefill_d128 fp8 : 142.0 -> 125.6 ms  (-11.5%)
        //   prefill_d128 bf16: unchanged (KY0_step_N=32 — gated out)
        //   prefill_d64  fp8 : unchanged (KY0_step_N=32 — gated out)
        //   prefill_d64  bf16: unchanged (KY0_step_N=64 — gated out, and
        //                       was incorrect under the runtime-gated variant)
        //
        // When kPageSize_ is known at compile time (host dispatched on the
        // runtime page_blk_size, see dispatch_variant<V> in
        // unified_attention.cpp) the gate becomes the *real* precondition
        // `KY0_step_N <= kPageSize`. Otherwise we fall back to the
        // conservative `<= 16` hedge that has to assume the smallest
        // production page size (vLLM/SGLang configure page_size ∈ {16, 32,
        // 64}; 16 is the smallest commonly configured value).
        //
        // The constexpr-page-size path unlocks the Tier-0 + Tier-2 fast
        // path on three additional prefill instances at their natural
        // page sizes (measured: sq=sk=75600 prefill, MI355, n=30 iters):
        //   prefill_d128 fp8   KY0_step_N=16 @ ps=32   119.0 → 111.5 ms (-6.3%)
        //                                              (was ON, now ON + strength-reduce)
        //   prefill_d128 bf16  KY0_step_N=32 @ ps=32   132.7 → 130.3 ms (-1.8%)
        //                                              (newly ON)
        //   prefill_d64  fp8   KY0_step_N=32 @ ps=32    80.9 →  68.1 ms (-15.8%)
        //                                              (newly ON — biggest win)
        //   prefill_d64  bf16  KY0_step_N=64 @ ps=64    74.4 →  73.4 ms (-1.3%)
        //                                              (newly ON; small win)
        constexpr index_t kKPageSizeCap = kHasCePageSize ? kPageSize : index_t{16};
        constexpr index_t kVPageSizeCap = kHasCePageSize ? kPageSize : index_t{16};
        // EXPERIMENT 2026-05: relax the 8-warp gate. The original measurement
        // showed 1-4 warp decode regressed 3-8%, but that was *before* the
        // halved-kBlockN bf16 change which (a) doubles the iter count, hence
        // doubles the per-tile page-table refresh count, and (b) lifts decode
        // occupancy from 1 CTA/CU to 3 CTAs/CU, multiplying the per-CU
        // contention on the per-lane block_tables_ptr_ vector loads. Both
        // shift the trade-off in favour of the Tier-2 LDS-cached path.
        constexpr bool kScalarPromoteKPageIdx =
            (KNRepeat >= 2) && (KY0_step_N <= kKPageSizeCap);
        constexpr bool kScalarPromoteVPageIdx =
            (VNRepeat >= 2) && (VY0_step_N <= kVPageSizeCap);

        // Warp-major ("contiguous-page") K rebase: each wave owns the contiguous
        // block [warp*tpw, (warp+1)*tpw) with tpw = kPageBlockSize / KLoadNumWarps.
        // When tpw <= page_size that block sits in ONE page, so we fold a per-wave
        // page base into the SRD (the per-wave analogue of the single-page rebase)
        // and read the page index straight from global — no Tier-2 LDS cache, no
        // per-lane block-table path. Defined here (ahead of the LDS-cache gate) so
        // the cache sizing can exclude K when it takes this path.
        constexpr index_t kTokensPerWarp = kPageBlockSize / KLoadNumWarps;
        constexpr bool kRebaseKSrdWarpMajor =
            Policy::kKNContigLoad && kIsPaged && kHasCePageSize &&
            (kTokensPerWarp <= kPageSize);

        // Multi-page K (KY0_step_N > page, e.g. ps16/ps32) currently drops to the
        // per-lane *global* fallback — the cliff seen at ps64->ps32. Two opt-in
        // paths route it through the LDS page-table cache instead:
        //   kFallbackUsesLdsK : keep per-lane structure, read from LDS not global
        //                       (isolates read-latency as the cause).
        //   kMultiPageDedupK  : resolve the G = KY0_step_N/page wave-uniform pages
        //                       per issue + per-lane select (cuts the 64 per-lane
        //                       reads to G scalar reads — the actual fix).
        constexpr bool kFallbackUsesLdsK = Policy::kKFallbackLds && kIsPaged &&
            kHasCePageSize && !kScalarPromoteKPageIdx && !kRebaseKSrdWarpMajor;
        constexpr bool kMultiPageDedupK = Policy::kKMultiPageDedup && kIsPaged &&
            kHasCePageSize && !kScalarPromoteKPageIdx && !kRebaseKSrdWarpMajor &&
            (KY0_step_N % kPageSize == 0);
        // Same fix on the V path: ps16/ps32 V also has VY0_step_N > page, so its
        // fallback was still doing per-lane *global* block_tables reads (the
        // residual VMEM/wait gap vs ps64). Route them through the shared LDS
        // cache too.
        constexpr bool kFallbackUsesLdsV = Policy::kKFallbackLds && kIsPaged &&
            kHasCePageSize && !kScalarPromoteVPageIdx;


        // Tier 2 — LDS-resident page-table cache.
        //
        // After Tier 0 the per-K/V-tile cost of resolving phys_page is a
        // single per-warp `s_load_dword` from global memory through the
        // scalar L1. The dependent address-comp chain only partially hides
        // its ~50-100 cycle latency — on long-context prefill those waits
        // still account for ~5-8% of total cycles (see the WAIT row of the
        // post-Tier-0 PC-sampling profile in BOTTLENECK_ANALYSIS.md).
        //
        // Tier 2 replaces them with a single cooperative bulk load at kernel
        // entry that stages this CTA's block_tables slice into LDS. Each
        // subsequent refresh_*_offsets call is then a one-cycle ds_read_b32
        // broadcast (every lane resolves to the same page-table index when
        // Tier 0 fires) instead of an s_load_dword + scoreboarding wait.
        //
        // This is also the optimisation Triton structurally cannot express:
        // Triton tensors model LDS as a static, statically-shaped tile and
        // dynamic per-thread indexing into LDS is not part of the language.
        // Lowering this to ds_read_b32 with a uniform per-warp address
        // requires a per-lane index expression, which only opens up at the
        // HIP/CK level.
        //
        // Capacity: kPageTableLdsEntries = 4096 × 4 B = 16 KiB. With the
        // per-split window load above, this bounds the *per-split* page
        // count rather than the absolute total. At page_size ∈ {16, 32, 64}
        // this covers sk_per_split up to {64 K, 128 K, 256 K} tokens —
        // since the wrapper splits to ~4× CU count, total sk supported is
        // roughly num_splits × that. (E.g. b=64 sk=128K page=16 → 4 splits
        // → 2000 pages/split, comfortably under 4096.) If a caller still
        // exceeds the cap we trap on the assert below rather than silently
        // miscompute; a runtime fallback was tried earlier and regresses
        // 30% because the compiler emits both refresh_*_offsets paths and
        // the resulting register pressure halves occupancy.
        // Warp-major K reads its (per-wave, uniform) page index straight from
        // global, so it never consumes the Tier-2 LDS cache; only count K when it
        // actually takes a cache-backed path. (V is unchanged in this prototype.)
        constexpr bool kKNeedsPageTableLds = (kScalarPromoteKPageIdx && !kRebaseKSrdWarpMajor) ||
                                             kFallbackUsesLdsK || kMultiPageDedupK;
        constexpr bool kVNeedsPageTableLds = kScalarPromoteVPageIdx || kFallbackUsesLdsV;
        constexpr bool kUsePageTableLds =
            kIsPaged && (kKNeedsPageTableLds || kVNeedsPageTableLds);
        constexpr index_t kPageTableLdsOffset =
            GetSmemSize() - GetPageTableLdsBytes();
        auto block_tables_lds = reinterpret_cast<int32_t*>(
            static_cast<char*>(smem_ptr) + kPageTableLdsOffset);

        // Split-KV correction: refresh_*_offsets indexes block_tables_lds by
        // i_base_page = (split_token_offset + …) / page_size, which is the
        // *absolute* page index within the batch (NOT relative to this split).
        // For prefill split_token_offset == 0 so absolute and relative indices
        // coincide; for split-KV decode (i_total_loops starts at
        // num_blocks_start > 0 on splits 1+), they diverge.
        //
        // Per-split window: each CTA only ever references pages in the
        // half-open range
        //     [split_start_page, split_end_page)
        //         = [⌊num_blocks_start · kPageBlockSize / page_size⌋,
        //            ⌈num_blocks       · kPageBlockSize / page_size⌉)
        //
        // We bulk-load just that window (a "split_window_pages" entry slice)
        // and shift refresh_*_offsets' lookup by split_start_page so the LDS
        // index stays in [0, split_window_pages). On the prefill path
        // (num_blocks_start == 0) split_start_page == 0 and this collapses to
        // the original absolute-indexed load. On split-KV (num_blocks_start >
        // 0) it both saves the bulk-load bytes for pages we skip past *and*
        // lets long-context decode (sk·kPageBlockSize/page_size > 4096) fit
        // under the cap, since the per-CTA window is bounded by
        // (total_kv_pages / num_splits) instead of the absolute total.
        const index_t split_start_page = static_cast<index_t>(
            (static_cast<long_index_t>(num_blocks_start) * kPageBlockSize) / page_size);
        const index_t split_end_page = static_cast<index_t>(
            (static_cast<long_index_t>(num_total_loop) * kPageBlockSize + page_size - 1) /
            page_size);
        const index_t split_window_pages = split_end_page - split_start_page;
        if constexpr (kUsePageTableLds)
        {
            assert(split_window_pages <= kPageTableLdsEntries);

            const index_t tid = get_thread_local_1d_id();
            for (index_t i = tid; i < split_window_pages; i += Problem::kBlockSize)
            {
                block_tables_lds[i] = block_tables_ptr_[block_table_offset + split_start_page + i];
            }
            // Each thread writes a strided subset of block_tables_lds[] and
            // subsequent refresh_*_offsets reads at i_base_page may be served
            // by a *different* lane's write (cross-lane LDS access). The
            // s_barrier below handles cross-wave ordering, but on single-warp
            // CTAs (TinyDecode, kBlockSize == warp_size) LLVM elides s_barrier
            // entirely — and with it the implicit lgkmcnt(0) drain that
            // commits this wave's ds_writes. Without an explicit drain the
            // refresh path then reads stale LDS. Adding `s_waitcnt lgkmcnt(0)`
            // is a no-op on the multi-warp tiers (the s_barrier carries it
            // implicitly) and load-bearing for single-warp tiers.
            s_waitcnt_lgkmcnt<0>();
            __builtin_amdgcn_s_barrier();
        }

        // Within-tile phys_page dedup gate (compile-time page geometry).
        //
        // The K/V tile is kPageBlockSize tokens wide and aligned to
        // kPageBlockSize. The scalar-promote path resolves phys_page once per
        // *issue* (KNRepeat ds_read_b32 broadcasts per tile), but multiple
        // issues frequently land in the SAME page — the compiler can't CSE the
        // reads because it can't prove `(base + i*Y0_step_N)/ps == base/ps` for
        // a runtime `base`. With a compile-time page_size the issue->page map is
        // a pure compile-time function in two provable regimes:
        //   (1) kPageBlockSize % kPageSize == 0  (page divides tile): tile_base
        //       is always a multiple of page_size, so issue i sits in relative
        //       page (i*Y0_step_N)/kPageSize.
        //   (2) kPageSize % kPageBlockSize == 0  (tile divides page): the whole
        //       kPageBlockSize-token tile fits in ONE page, so every issue
        //       shares relative page 0, regardless of tile_base alignment.
        // In both regimes we resolve phys_page once per distinct page and reuse
        // it, collapsing the B2 K-mem straggler's ds_read / readfirstlane count
        // (to a single read at ps >= kPageBlockSize). The dedup needs a
        // compile-time page_size, so it is gated on kHasCePageSize; the runtime
        // page_size scalar-promote path keeps the original per-issue read, and
        // any exotic (non-dividing) page_size falls back to it too.
        constexpr bool kDedupPages =
            kHasCePageSize &&
            (kPageBlockSize % kPageSize == 0 || kPageSize % kPageBlockSize == 0);

        // Single-page SRD-rebase fast path.
        //
        // When the whole K/V tile lives in one physical page
        // (kPageSize % kPageBlockSize == 0 — the tile is kPageBlockSize-aligned
        // and the page is a multiple of the tile, so it never straddles a page
        // boundary), the per-issue byte offset
        //   (phys_page*kPageSize + within_page) * row_stride
        // splits cleanly into
        //   U(tile)  = (phys_page*kPageSize + tile_base_token
        //               - base_page*kPageSize) * row_stride   [wave-uniform]
        //   L(i,lane)= (i*Y0_step_N + thread_n_pos) * row_stride [loop-invariant]
        // with offset(i) = U(tile) + L(i,lane). We fold U into the buffer
        // descriptor (SRD) base — rebased once per tile — and leave only L in
        // the scatter offset array. Because the pipeline already rebuilds the
        // window + SRD every iteration (make_tile_scatter_gather + init_raw
        // below), the rebase adds *no* extra SRD construction; it only swaps a
        // different (wave-uniform) base into make_wave_buffer_resource. The
        // payoff: L is loop-invariant, so the per-lane 64-bit multiply-add that
        // the ATT profile attributes to the ADDR phase disappears from the tile
        // loop. L is also bounded by one tile, so it can never overflow int32 —
        // the rebased path never needs the _long load variant.
        //
        // This is the per-tile analogue of the legacy PageBlockNavigator
        // uniform-base rebase. The earlier per-*issue* SRD rebase was dropped
        // (see the K/V dispatch comment) for SGPR pressure under the old
        // ping-pong pipeline; the per-tile form here piggybacks on the existing
        // per-iteration SRD build, so it does not add that pressure.
        //
        // Gated per-K/V to the scalar-promote regime AND single-page geometry.
        // The single-page geometry (kPageSize % kPageBlockSize == 0) is what lets
        // ONE wave-uniform base address the whole tile; scalar-promote
        // (KNRepeat>=2 && KY0_step_N<=page) is the regime in which the residual
        // per-lane offset is proven to stay within that tile. Outside it the
        // issue layout is wider (notably V's transposed load, e.g. bf16 d128
        // where scalar-promote_V is off) and a single base does not cover every
        // issue — broadening the gate to all single-page faults there — so those
        // keep the validated per-lane scatter, as do multi-page tiles
        // (ps < kPageBlockSize, e.g. d128 @ ps16). K and V are gated
        // independently because their geometries (and thus scalar-promote) differ.
        // Single-page SRD-rebase eligibility. The *geometric* precondition is only
        // "the whole KV tile sits in one page": kPageSize % kPageBlockSize == 0
        // (tile-aligned starts) means [tile_base, tile_base+kPageBlockSize) never
        // straddles a page, so ONE wave-uniform base addresses the tile.
        //
        // Historically the rebase was gated behind kScalarPromote* (:= NRepeat>=2
        // && step<=page). That is the SEPARATE "multiple issues share a page index"
        // optimization, and conflating the two wrongly excluded the trivial
        // NRepeat==1 single-issue tile -- bf16/fp16 d128 @ ps128 (tile N=32 in a
        // 128-page, NRepeat=1) -- forcing it onto the per-lane multi-page fallback
        // (~500 addr cyc/warp in the ATT trace; the contiguous->paged cliff). A
        // single issue spanning step<=page is trivially within one page, so admit
        // it explicitly. Kept minimal: only the NRepeat==1 arm is added, so the
        // validated NRepeat>=2 behaviour is untouched.
        constexpr bool kSinglePageGeom =
            kHasCePageSize && (kPageSize % kPageBlockSize == 0);
        constexpr bool kRebaseKSrd =
            kSinglePageGeom &&
            (kScalarPromoteKPageIdx || (KNRepeat == 1 && KY0_step_N <= kPageSize));
        constexpr bool kRebaseVSrd =
            kSinglePageGeom &&
            (kScalarPromoteVPageIdx || (VNRepeat == 1 && VY0_step_N <= kPageSize));


        // Wave-uniform per-tile base offsets (in elements) folded into the SRD
        // base at window construction; written by refresh_*_offsets.
        long_index_t k_srd_base_offset = 0;
        long_index_t v_srd_base_offset = 0;

        // Cross-stagger phys_page carry. K runs exactly one tile ahead of V and
        // they share the same logical->physical page map (same block_tables),
        // so V can reuse the phys_page K already read + broadcast instead of
        // issuing its own block_tables_lds read + readfirstlane -- the dominant
        // WG1 addr-phase stall in the paged path. Keyed by tile parity (R=2):
        // refresh_k writes ring[tile&1], refresh_v reads ring[tile&1]. Since K
        // is at most one tile ahead, the slot V reads (tile N) is only rewritten
        // for tile N+2, strictly after V has consumed it; and because the ring
        // is keyed by tile (not by call order) the value stays valid even on the
        // loop tail where refresh_k stops early. Only valid when both K and V
        // take the single-page rebase path AND every wave refreshes both tiles
        // (cooperative load); under the WG-specialized load roles a wave sees
        // only one of K/V, so V must still read its own page.
        // Disabled under warp-major K: K then resolves a PER-WAVE phys_page that
        // does not match V's tile-uniform single-page base, so the cross-stagger
        // carry would feed V the wrong page.
        constexpr bool kCarryKVPhys = kRebaseKSrd && kRebaseVSrd && !kRebaseKSrdWarpMajor &&
                                      !Policy::kFA4WG1LoadsK && !Policy::kFA4WG0LoadsV;
        int32_t kv_phys_ring0 = 0;
        int32_t kv_phys_ring1 = 0;

        auto refresh_k_offsets = [&](index_t k_tile_idx, auto is_init) {
            (void)is_init;
            if constexpr(!kIsPaged)
            {
                // Contiguous (THD) K: no page table. The logical token index is
                // the physical row directly (the kernel folded the per-sequence
                // KV start into the K base pointer), so the offset collapses to
                // logical_token * row_stride — no block_tables, no page split.
                if constexpr(decltype(is_init)::value)
                {
                    static_for<0, KNRepeat, 1>{}([&](auto i) {
                        const index_t logical_token =
                            split_token_offset + k_tile_idx * kPageBlockSize + k_thread_n_pos +
                            static_cast<index_t>(i.value) * KY0_step_N;
                        k_page_offsets(i) =
                            static_cast<long_index_t>(logical_token) * k_row_stride;
                    });
                }
                else
                {
                    // Steady state: k_tile_idx only ever advances by +1/tile, so
                    // every repeat's byte offset moves by the SAME loop-invariant
                    // kPageBlockSize*row_stride. Incrementing kills the per-tile
                    // 64-bit v_mad_i64_i32 + most of the v_lshl_add address chain
                    // that the ATT trace showed exposed in the bf16 matrix phase.
                    const long_index_t k_tile_stride =
                        static_cast<long_index_t>(kPageBlockSize) *
                        static_cast<long_index_t>(k_row_stride);
                    static_for<0, KNRepeat, 1>{}(
                        [&](auto i) { k_page_offsets(i) += k_tile_stride; });
                }
            }
            else if constexpr(kRebaseKSrdWarpMajor)
            {
                // Per-wave page rebase (warp-major load). Wave `warp` owns the
                // contiguous block [warp*tpw, (warp+1)*tpw) ⊆ one page, so fold the
                // wave's page base into the SRD and keep the per-lane scatter
                // (lanegroup + issue*step) loop-invariant — the multi-page analogue
                // of kRebaseKSrd, with a base that differs per wave.
                // warp_n_base must be wave-UNIFORM (it feeds the SRD base, an SGPR
                // operand of buffer_load). k_thread_n_pos = warp*tpw + lanegroup is
                // per-lane, but warp*tpw is constant within the wave, so promote it
                // through readfirstlane; lane_within keeps the per-lane remainder.
                const index_t warp_n_base =
                    __builtin_amdgcn_readfirstlane(k_thread_n_pos / kTokensPerWarp) *
                    kTokensPerWarp; // = warp*tpw, scalar
                const index_t lane_within = k_thread_n_pos - warp_n_base; // within-page lane pos
                const index_t wave_base_token =
                    split_token_offset + k_tile_idx * kPageBlockSize + warp_n_base;
                const int32_t base_page =
                    __builtin_amdgcn_readfirstlane(wave_base_token / kPageSize);
                // Read the page index straight from global. Under warp-major each
                // wave needs exactly ONE phys_page per tile (the per-lane spread is
                // within-page), so this is a single wave-uniform scalar load — not
                // the per-lane vector gather the Tier-2 LDS cache was built to
                // avoid. Reading directly means no LDS cache, hence no
                // kPageTableLdsEntries cap on context length.
                const int32_t phys_page = __builtin_amdgcn_readfirstlane(
                    block_tables_ptr_[block_table_offset + base_page]);
                k_srd_base_offset =
                    (static_cast<long_index_t>(phys_page) * kPageSize +
                     (wave_base_token - static_cast<long_index_t>(base_page) * kPageSize)) *
                    k_row_stride;
                static_for<0, KNRepeat, 1>{}([&](auto i) {
                    constexpr index_t ii = i.value;
                    k_page_offsets(i) =
                        (static_cast<long_index_t>(ii) * KY0_step_N + lane_within) * k_row_stride;
                });
            }
            else if constexpr(kRebaseKSrd)
            {
                // Wave-uniform: the element offset of the tile's first token
                // within the K pool. Folded into the SRD base (see window
                // construction below).
                const index_t tile_base_token =
                    split_token_offset + k_tile_idx * kPageBlockSize;
                const int32_t base_page =
                    __builtin_amdgcn_readfirstlane(tile_base_token / kPageSize);
                // Must be wave-uniform: it feeds the SRD base (an SGPR operand
                // of buffer_load). The LDS read alone does not prove uniformity
                // to the backend, so force it through readfirstlane.
                const int32_t phys_page = __builtin_amdgcn_readfirstlane(
                    block_tables_lds[base_page - split_start_page]);
                // Publish for the staggered V refresh (see kCarryKVPhys). The
                // parity branch is on a wave-uniform value, so it lowers to a
                // scalar select and the ring stays in SGPRs.
                if constexpr(kCarryKVPhys)
                {
                    if(k_tile_idx & 1)
                        kv_phys_ring1 = phys_page;
                    else
                        kv_phys_ring0 = phys_page;
                }
                k_srd_base_offset =
                    (static_cast<long_index_t>(phys_page) * kPageSize +
                     (tile_base_token - static_cast<long_index_t>(base_page) * kPageSize)) *
                    k_row_stride;
                // Loop-invariant per-lane within-tile offset — the only term
                // left in the scatter array. The compiler hoists it out of the
                // tile loop.
                static_for<0, KNRepeat, 1>{}([&](auto i) {
                    constexpr index_t ii = i.value;
                    k_page_offsets(i) =
                        (static_cast<long_index_t>(ii) * KY0_step_N + k_thread_n_pos) *
                        k_row_stride;
                });
            }
            else if constexpr(kScalarPromoteKPageIdx && kDedupPages)
            {
                // Reached only for MULTI-page tiles — single-page took the rebase
                // branch above — so the tile spans kPageBlockSize/kPageSize pages
                // and issue i sits in relative page (i*KY0_step_N)/kPageSize.
                // One uniform readfirstlane for the tile's first page; the
                // per-issue page index is then base_page + a compile-time
                // relative offset, so no further readfirstlane is emitted.
                const index_t tile_base_token =
                    split_token_offset + k_tile_idx * kPageBlockSize;
                const int32_t base_page =
                    __builtin_amdgcn_readfirstlane(tile_base_token / kPageSize);
                // Shift by split_start_page to convert absolute -> window index
                // (see "Per-split window" comment above the cache load).
                int32_t phys_page = block_tables_lds[base_page - split_start_page];
                static_for<0, KNRepeat, 1>{}([&](auto i) {
                    constexpr index_t ii = i.value;
                    constexpr index_t grp = (ii * KY0_step_N) / kPageSize;
                    // Re-read phys_page only when this issue crosses into a new
                    // page (a compile-time decision); otherwise reuse the value
                    // already in the VGPR.
                    if constexpr(ii > 0)
                    {
                        constexpr index_t grp_prev = ((ii - 1) * KY0_step_N) / kPageSize;
                        if constexpr(grp != grp_prev)
                            phys_page =
                                block_tables_lds[base_page + grp - split_start_page];
                    }
                    const index_t logical_token =
                        tile_base_token + ii * KY0_step_N + k_thread_n_pos;
                    const index_t within_page =
                        logical_token - (base_page + grp) * kPageSize;
                    k_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * kPageSize + within_page) *
                        k_row_stride;
                });
            }
            else if constexpr(kScalarPromoteKPageIdx)
            {
                static_for<0, KNRepeat, 1>{}([&](auto i) {
                    // Compute the uniform per-`i` base in scalar; force the
                    // resulting page-table index into an SGPR. Tier 2 reads
                    // the phys_page from the LDS cache populated above (one
                    // ds_read_b32 broadcast per warp); Tier 0 falls back to
                    // the s_load_dword path when the cache is absent.
                    const index_t i_base_token = split_token_offset +
                                                 k_tile_idx * kPageBlockSize +
                                                 static_cast<index_t>(i.value) * KY0_step_N;
                    const int32_t i_base_page  = __builtin_amdgcn_readfirstlane(
                        i_base_token / page_size);
                    const int32_t phys_page    = block_tables_lds[i_base_page - split_start_page];
                    const index_t logical_token = i_base_token + k_thread_n_pos;
                    const index_t within_page   = logical_token - i_base_page * page_size;
                    k_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * page_size + within_page) *
                        k_row_stride;
                });
            }
            else if constexpr(kMultiPageDedupK)
            {
                // MULTI-page tile, per-lane spread KY0_step_N > page_size (e.g.
                // ps16/ps32). Within one issue the spread still covers only
                // G = KY0_step_N/page_size distinct, page-aligned pages, so
                // resolve those G wave-uniform phys_pages from the LDS cache (one
                // readfirstlane each) and select per-lane by k_thread_n_pos/page,
                // instead of 64 per-lane *global* block_tables reads. The tile is
                // page-aligned (kPageBlockSize % page == 0) and each issue base
                // (tile_base + i*KY0_step_N) is page-aligned (KY0_step_N % page ==
                // 0), so g_lane = k_thread_n_pos/page lands the lane in
                // pp[g_lane] exactly.
                constexpr index_t G = KY0_step_N / kPageSize;
                const index_t tile_base_token =
                    split_token_offset + k_tile_idx * kPageBlockSize;
                static_for<0, KNRepeat, 1>{}([&](auto i) {
                    constexpr index_t ii = i.value;
                    const index_t i_base_token = tile_base_token + ii * KY0_step_N;
                    const int32_t i_base_page =
                        __builtin_amdgcn_readfirstlane(i_base_token / kPageSize);
                    int32_t pp[G];
                    static_for<0, G, 1>{}([&](auto g) {
                        pp[g.value] =
                            block_tables_lds[i_base_page + g.value - split_start_page];
                    });
                    const index_t g_lane      = k_thread_n_pos / kPageSize;   // [0, G)
                    const index_t within_page = k_thread_n_pos - g_lane * kPageSize;
                    int32_t phys_page = pp[0];
                    static_for<1, G, 1>{}([&](auto g) {
                        if(g_lane == static_cast<index_t>(g.value))
                            phys_page = pp[g.value];
                    });
                    k_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * kPageSize + within_page) *
                        k_row_stride;
                });
            }
            else
            {
                static_for<0, KNRepeat, 1>{}([&](auto i) {
                    // Byte-identical to the pre-optimisation path, except the
                    // phys_page read is routed to the LDS cache under the
                    // kFallbackUsesLdsK probe (per-lane structure unchanged).
                    const index_t logical_token = split_token_offset +
                                                  k_tile_idx * kPageBlockSize + k_thread_n_pos +
                                                  static_cast<index_t>(i.value) * KY0_step_N;
                    const index_t logical_page  = logical_token / page_size;
                    const index_t within_page   = logical_token - logical_page * page_size;
                    const index_t phys_page =
                        kFallbackUsesLdsK
                            ? block_tables_lds[logical_page - split_start_page]
                            : block_tables_ptr_[block_table_offset + logical_page];
                    k_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * page_size + within_page) *
                        k_row_stride;
                });
            }
        };
        auto refresh_v_offsets = [&](index_t v_tile_idx, auto is_init) {
            (void)is_init;
            if constexpr(!kIsPaged)
            {
                // Contiguous (THD) V — see refresh_k_offsets.
                if constexpr(decltype(is_init)::value)
                {
                    static_for<0, VNRepeat, 1>{}([&](auto i) {
                        const index_t logical_token =
                            split_token_offset + v_tile_idx * kPageBlockSize + v_thread_n_pos +
                            static_cast<index_t>(i.value) * VY0_step_N;
                        v_page_offsets(i) =
                            static_cast<long_index_t>(logical_token) * v_row_stride;
                    });
                }
                else
                {
                    const long_index_t v_tile_stride =
                        static_cast<long_index_t>(kPageBlockSize) *
                        static_cast<long_index_t>(v_row_stride);
                    static_for<0, VNRepeat, 1>{}(
                        [&](auto i) { v_page_offsets(i) += v_tile_stride; });
                }
            }
            else if constexpr(kRebaseVSrd)
            {
                // Single-page SRD rebase — see refresh_k_offsets / the
                // kRebaseKSrd comment above for the U + L decomposition.
                const index_t tile_base_token =
                    split_token_offset + v_tile_idx * kPageBlockSize;
                const int32_t base_page =
                    __builtin_amdgcn_readfirstlane(tile_base_token / kPageSize);
                // Wave-uniform (SRD base SGPR operand) — see refresh_k_offsets.
                // Reuse the phys_page K already broadcast for this same logical
                // tile (kCarryKVPhys) to elide V's own block-table LDS read +
                // readfirstlane; fall back to the read under WG-specialized loads.
                const int32_t phys_page = [&]() -> int32_t {
                    if constexpr(kCarryKVPhys)
                        return (v_tile_idx & 1) ? kv_phys_ring1 : kv_phys_ring0;
                    else
                        return __builtin_amdgcn_readfirstlane(
                            block_tables_lds[base_page - split_start_page]);
                }();
                v_srd_base_offset =
                    (static_cast<long_index_t>(phys_page) * kPageSize +
                     (tile_base_token - static_cast<long_index_t>(base_page) * kPageSize)) *
                    v_row_stride;
                static_for<0, VNRepeat, 1>{}([&](auto i) {
                    constexpr index_t ii = i.value;
                    v_page_offsets(i) =
                        (static_cast<long_index_t>(ii) * VY0_step_N + v_thread_n_pos) *
                        v_row_stride;
                });
            }
            else if constexpr(kScalarPromoteVPageIdx && kDedupPages)
            {
                // Multi-page only (single-page took the rebase branch above);
                // see refresh_k_offsets for the dedup rationale.
                const index_t tile_base_token =
                    split_token_offset + v_tile_idx * kPageBlockSize;
                const int32_t base_page =
                    __builtin_amdgcn_readfirstlane(tile_base_token / kPageSize);
                int32_t phys_page = block_tables_lds[base_page - split_start_page];
                static_for<0, VNRepeat, 1>{}([&](auto i) {
                    constexpr index_t ii = i.value;
                    constexpr index_t grp = (ii * VY0_step_N) / kPageSize;
                    if constexpr(ii > 0)
                    {
                        constexpr index_t grp_prev = ((ii - 1) * VY0_step_N) / kPageSize;
                        if constexpr(grp != grp_prev)
                            phys_page =
                                block_tables_lds[base_page + grp - split_start_page];
                    }
                    const index_t logical_token =
                        tile_base_token + ii * VY0_step_N + v_thread_n_pos;
                    const index_t within_page =
                        logical_token - (base_page + grp) * kPageSize;
                    v_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * kPageSize + within_page) *
                        v_row_stride;
                });
            }
            else if constexpr(kScalarPromoteVPageIdx)
            {
                static_for<0, VNRepeat, 1>{}([&](auto i) {
                    const index_t i_base_token = split_token_offset +
                                                 v_tile_idx * kPageBlockSize +
                                                 static_cast<index_t>(i.value) * VY0_step_N;
                    const int32_t i_base_page  = __builtin_amdgcn_readfirstlane(
                        i_base_token / page_size);
                    // Window-relative index; see K-path comment for rationale.
                    const int32_t phys_page    = block_tables_lds[i_base_page - split_start_page];
                    const index_t logical_token = i_base_token + v_thread_n_pos;
                    const index_t within_page   = logical_token - i_base_page * page_size;
                    v_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * page_size + within_page) *
                        v_row_stride;
                });
            }
            else
            {
                static_for<0, VNRepeat, 1>{}([&](auto i) {
                    const index_t logical_token = split_token_offset +
                                                  v_tile_idx * kPageBlockSize + v_thread_n_pos +
                                                  static_cast<index_t>(i.value) * VY0_step_N;
                    const index_t logical_page  = logical_token / page_size;
                    const index_t within_page   = logical_token - logical_page * page_size;
                    const index_t phys_page =
                        kFallbackUsesLdsV
                            ? block_tables_lds[logical_page - split_start_page]
                            : block_tables_ptr_[block_table_offset + logical_page];
                    v_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * page_size + within_page) *
                        v_row_stride;
                });
            }
        };

        refresh_k_offsets(k_block_idx, std::true_type{});
        refresh_v_offsets(v_block_idx, std::true_type{});

        auto k_view = k_dram_block_window_tmp.get_bottom_tensor_view();
        auto v_view = v_dram_block_window_tmp.get_bottom_tensor_view();

        // Single-page SRD rebase: fold the wave-uniform per-tile page offset
        // into the buffer (SRD) base instead of the per-lane scatter offsets.
        // The SRD is rebased once per tile (see K_mem_load / V_mem_load), so
        // the scatter array carries only the loop-invariant within-tile offset
        // (see kRebaseKSrd above). We stash the original pool base here so the
        // per-tile rebase can recompute base = pool_base + U(tile); the first
        // tile's base is set right below before the window is built.
        //
        // The buffer_size_ bound is intentionally left at the full pool extent:
        // every loaded token lives inside a valid physical page (the tile never
        // straddles a page in this regime), so no read exceeds the pool,
        // exactly as on the per-lane path where the voffset is always in bounds
        // and the softmax mask handles past-seqlen tokens.
        [[maybe_unused]] auto* const k_pool_base = k_view.get_buffer_view().p_data_;
        [[maybe_unused]] auto* const v_pool_base = v_view.get_buffer_view().p_data_;
        if constexpr(kRebaseKSrd || kRebaseKSrdWarpMajor)
            k_view.get_buffer_view().p_data_ = k_pool_base + k_srd_base_offset;
        if constexpr(kRebaseVSrd)
            v_view.get_buffer_view().p_data_ = v_pool_base + v_srd_base_offset;

        auto k_dram_window =
            make_tile_scatter_gather(k_view,
                                     k_dram_block_window_tmp.get_window_lengths(),
                                     {0, 0},
                                     k_dist,
                                     k_page_offsets);
        k_dram_window.init_raw();

        // For the single-page rebase regime the V per-lane scatter array is
        // bit-identical to K's: same kv_cache strides (k_row_stride ==
        // v_row_stride), same DRAM distribution (GetAlignmentK == GetAlignmentV
        // for fp8, so KY0_step_N == VY0_step_N and the thread positions match),
        // and the only per-tile divergence is the wave-uniform SRD base
        // (k/v_srd_base_offset), already folded into the view above. Feed the
        // SAME loop-invariant offset array to both windows so the backend can
        // coalesce the otherwise-duplicated page_idx_ storage. Gated on
        // KNRepeat == VNRepeat so the array types match (always true here since
        // both rebase flags imply the shared fp8 geometry).
        // Not shareable under warp-major K: K's scatter array uses the per-wave
        // within-page layout, V keeps the default interleaved layout.
        constexpr bool kShareKVScatter =
            kRebaseKSrd && kRebaseVSrd && !kRebaseKSrdWarpMajor && (KNRepeat == VNRepeat);
        auto v_dram_window = make_tile_scatter_gather(
            v_view,
            v_dram_block_window_tmp.get_window_lengths(),
            {0, 0},
            v_dist,
            [&]() -> const auto& {
                if constexpr(kShareKVScatter)
                    return k_page_offsets;
                else
                    return v_page_offsets;
            }());
        v_dram_window.init_raw();

        // prefetch K tile
        constexpr index_t k0_loops = 1;
        constexpr index_t k1_loops = 1;
        static_assert(1 == k0_loops);
        static_assert(1 == k1_loops);
        // static_assert(kPageBlockSize == kHeadDimPadded);

        constexpr index_t NumWarpGroups = Problem::kBlockSize / Policy::NumThreadPerWarpGroup;
        static_assert(NumWarpGroups == 1 || NumWarpGroups == 2);

        // Conditional (skipped) online-softmax rescale applies only to the
        // 2-warp-group *prefill* pipeline, which is VALU/rescale-bound. The
        // single-warp-group *decode* path (NumWarpGroups==1) is memory-bound,
        // its o_acc is small, and the per-tile ballot+branch overhead is not
        // recovered (measured ~+2% regression) — so decode keeps the
        // always-rescale path. Gated at compile time, so each instance lowers
        // to exactly one path with no runtime cost.
        constexpr bool kCondRescale = (CONDITIONAL_RESCALE != 0) && (NumWarpGroups == 2);

        // FA4 matrix‖softmax warp-group overlap (see the FA4 pipeline notes at
        // the top of this file). Enabled for the 2-warp-group prefill path; it
        // is the only 2-WG pipeline now. The constraint is the FP8 P-tile
        // QK-C→PV-A relayout inside fmha_alu1: FA4 splits the 8 warps into two
        // groups that run *different* phases at once, so any *block-wide*
        // s_barrier inside a single group's softmax phase deadlocks (the matrix
        // group never reaches it).
        //
        // Crucially the FP8 relayout has two strategies (see fmha_alu1): the
        // 16x16x32 m16 tier takes the block-wide LDS-roundtrip path (strategy
        // B — two s_barriers, NOT FA4-safe), but every 32x32x16 tier (all
        // prefill, decode_m{32,64,128}) takes the *within-wave* permute
        // (strategy A: permlane32_swap on gfx950 / ds_bpermute on gfx942 —
        // zero LDS traffic, zero barriers). The within-wave path adds nothing
        // to the per-phase barrier balance, so FP8 prefill is FA4-safe exactly
        // when Gemm1WarpTile is 32x32x16. m16 (16x16x32) is single-warp-group
        // anyway (NumWarpGroups==1), so the NumWarpGroups==2 guard already
        // excludes the strategy-B case; the explicit tile check below documents
        // the invariant and keeps any future 2-WG 16x16x32 instance on the
        // baseline.
        using Gemm1WarpTileFA4 = typename UnifiedAttentionShape::Gemm1WarpTile;
        // Barrier-free QK-C->PV-A FP8 relayout is available for two 32x32 tiles:
        //   K=16 -> strategy A (within-wave permlane32_swap, see fmha_alu1)
        //   K=64 -> strategy C (cvt-only; QK-C and PV-A layouts already match
        //           under the wide v_mfma_f32_32x32x64 MMA, like the ASM kernel)
        // Both are FA4-safe (no block barrier inside a single group's softmax).
        constexpr bool kFP8RelayoutWithinWave =
            (Gemm1WarpTileFA4::at(number<0>{}) == 32) &&
            (Gemm1WarpTileFA4::at(number<1>{}) == 32) &&
            (Gemm1WarpTileFA4::at(number<2>{}) == 16 ||
             Gemm1WarpTileFA4::at(number<2>{}) == 64);
        // FA4 is now the ONLY 2-warp-group prefill pipeline; the legacy
        // ping-pong baseline was removed. Every compiled 2-WG instance uses
        // the within-wave FP8 P relayout (32x32x16 Gemm1 tile), so kFA4 is
        // unconditionally true for NumWarpGroups==2. The static_assert pins
        // that invariant: a hypothetical future 2-WG 16x16x32 instance would
        // have no 2-WG path left, so fail the build loudly rather than run an
        // empty main loop.
        constexpr bool kFA4 = (NumWarpGroups == 2) &&
                              (!std::is_same_v<PDataType, fp8_t> || kFP8RelayoutWithinWave);
        static_assert(NumWarpGroups == 1 || kFA4,
                      "2-warp-group UA instances must be FA4-capable (32x32x16 FP8 P "
                      "relayout); the legacy ping-pong baseline was removed.");

        [[maybe_unused]] auto print_dist_tensor = [&](const auto& dist_tensor, const char* name) {
            printf("[POYENC] %s (size=%d): %5.2f",
                   name,
                   decltype(dist_tensor.thread_buf_)::size(),
                   ck_tile::type_convert<float>(dist_tensor.thread_buf_[0]));
            static_for<1, decltype(dist_tensor.thread_buf_)::size(), 1>{}([&](auto i) {
                printf(", %5.2f", ck_tile::type_convert<float>(dist_tensor.thread_buf_[i]));
            });
            printf("\n");
        };

        [[maybe_unused]] auto print_lds = [&](auto lds_tile_window, const char* name) {
            const auto num_rows = lds_tile_window.get_window_lengths().at(number<0>{});
            const auto num_cols = lds_tile_window.get_window_lengths().at(number<1>{});

            auto desc = lds_tile_window.get_bottom_tensor_view().desc_;
            auto data = lds_tile_window.get_bottom_tensor_view().buf_.p_data_;

            if constexpr(true || num_rows < num_cols)
            {
                for(int row = 0; row < num_rows; ++row)
                {
                    int offset = desc.calculate_offset(make_tuple(row, 0));
                    printf("[DEVICE] %s[%3d] = %5.2f",
                           name,
                           row,
                           ck_tile::type_convert<float>(data[offset]));
                    for(int col = 1; col < num_cols; ++col)
                    {
                        printf(", ");
                        offset = desc.calculate_offset(make_tuple(row, col));
                        printf("%5.2f", ck_tile::type_convert<float>(data[offset]));
                    }
                    printf("\n");
                }
            }
            else
            {
                for(int col = 0; col < num_cols; ++col)
                {
                    int offset = desc.calculate_offset(make_tuple(0, col));
                    printf("[DEVICE] %s[%3d] = %5.2f",
                           name,
                           col,
                           ck_tile::type_convert<float>(data[offset]));
                    for(int row = 1; row < num_rows; ++row)
                    {
                        printf(", ");
                        offset = desc.calculate_offset(make_tuple(row, col));
                        printf("%5.2f", ck_tile::type_convert<float>(data[offset]));
                    }
                    printf("\n");
                }
            }
        };

        [[maybe_unused]] auto print_lds_1d = [&](auto lds_tile_window, const char* name) {
            const auto num_elems = lds_tile_window.get_window_lengths().at(number<0>{});

            auto desc = lds_tile_window.get_bottom_tensor_view().desc_;
            auto data = lds_tile_window.get_bottom_tensor_view().buf_.p_data_;

            int offset = desc.calculate_offset(make_tuple(0));
            printf("[DEVICE] %s = %5.2f", name, ck_tile::type_convert<float>(data[offset]));
            for(int e = 1; e < num_elems; ++e)
            {
                printf(", ");
                offset = desc.calculate_offset(make_tuple(e));
                printf("%5.2f", ck_tile::type_convert<float>(data[offset]));
            }
            printf("\n");
        };

        // K_mem_su_ld_insts = 1 for 32 x 128
        // V_mem_su_ld_insts = 1 for 128 x 32
        constexpr int K_mem_su_ld_insts = k_dram_window.get_num_of_access();
        constexpr int V_mem_su_ld_insts = v_dram_window.get_num_of_access();

        // Page block index tracking
        // const index_t kv_page_size_in_blocks =
        //     PageSize / kPageBlockSize;
        // index_t kv_block_idx = 0;
        // only for block 0 and thread
        if(blockIdx.x == 0 && threadIdx.x == 0) {}

        // Pass-2: page indirection lives in page_offsets, not in the SRD. We
        // refresh the per-iter offsets table and push it to the window via
        // update_page_idx(); the SRD itself stays put (no init_raw per iter).
        //
        // Two load paths, dispatched on the runtime overflow flag:
        //   - false: `async_load_tile_raw` → `buffer_load_dword_lds` with a
        //     wave-uniform 4 GB-capped SRD. Faster, but per-lane voffsets
        //     are int32 so the path is only correct while
        //     `num_blocks * page_size * row_stride * sizeof(T) ≤ INT32_MAX`.
        //   - true: `async_load_tile_raw_long` → `global_load_lds_dwordx*`
        //     with per-lane 64-bit base pointers, lifting the 4 GB limit
        //     at the cost of lower throughput.
        // The branch is on a wave-uniform value, so no execution divergence.
        //
        // We tried a third "per-issue SRD rebase" path
        // (`async_load_tile_raw_rebased`: buffer_load_dword_lds with a
        // per-issue SRD whose 48-bit base absorbs the wave-uniform page
        // offset, valid when WaveSpanInN ≤ runtime page_size). It was
        // correct on every big-cache decode shape tested but came out at
        // best tied with the long path and at worst ~6% slower (e.g.
        // b=1 sk=1M d=64: 2.46 ms vs 2.32 ms; b=8 sk=200k d=128: 2.12 ms
        // vs 2.02 ms — see git log for the full table). These workloads
        // are compute / softmax bound, not K/V-load bandwidth bound, so
        // the buffer_load vs global_load_lds throughput edge never
        // materialises, while per-issue SRD construction adds real SGPR
        // pressure. The rebased helper has been removed to keep the
        // dispatch (and emitted kernel size) minimal.
        constexpr index_t KWaveSpanInN =
            (KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<1>{}] - 1) *
                KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<2>{}] +
            1;
        (void)KWaveSpanInN; // currently informational only

        // The post-load refresh prepares page_offsets for the NEXT K/V tile.
        // On the last tile of the (split-KV per-split) loop the next load is
        // never issued, but an unconditional refresh would still read
        // `block_tables[block_table_offset + (last_relative_tile + 1)]` —
        // for the *last* split that's one past the last valid logical_page
        // for this seq, i.e. an OOB read. When block_tables happens to be
        // the last allocation in a memory page that read faults; the PyTorch
        // caching allocator hides the bug for small workloads but reproduces
        // reliably once a workload deep-copies enough small `block_tables`
        // tensors (>~30 distinct copies on MI300/MI355) to spread allocations
        // across unmapped page boundaries.
        //
        // Gating refresh on the *per-split* iteration count
        // (num_total_loop - num_blocks_start) leaves the page_offsets table
        // stale on the final iter (harmless — no subsequent load consumes it)
        // and avoids the OOB read.
        //
        // Note: k_block_idx / v_block_idx are 0-based *relative to this
        // split*, while num_total_loop is the absolute end index. The
        // relative iteration count is therefore num_total_loop minus
        // num_blocks_start.
        const index_t num_iters_per_split = num_total_loop - num_blocks_start;
        auto K_mem_load = [&](auto k_lds_write_idx) {
            // K async load. With kFA4WG1LoadsK only WG1's 4 waves issue it (its
            // KLoadNumWarps==4 layout fills the full shared K tile) and WG0 reads K
            // from shared LDS; with the cooperative load (kFA4WG1LoadsK=false) all 8
            // waves load their 1/8 shard. Cooperative keeps each wave's load shard
            // (and addressing live-set) small enough to avoid the kv128 VGPR spills.
            if(k_load_active)
            {
                if(cache_ptr_int32_overflow_possible)
                    async_load_tile_raw_long(k_lds_window_store(k_lds_write_idx), k_dram_window);
                else
                    async_load_tile_raw(k_lds_window_store(k_lds_write_idx), k_dram_window);
            }
            k_block_idx++;
            // Only the K-loading warp group(s) refresh K offsets; k_block_idx stays
            // uniform across all waves so loop control and buffer parity never diverge.
            if(k_load_active && k_block_idx < num_iters_per_split)
            {
                refresh_k_offsets(k_block_idx, std::false_type{});
                if constexpr(kRebaseKSrd || kRebaseKSrdWarpMajor)
                    // Per-tile SRD rebase: the scatter offsets (L) are
                    // loop-invariant, so only the (per-wave) base moves.
                    k_dram_window.rebase_buffer_base(k_pool_base + k_srd_base_offset);
                else
                    k_dram_window.update_page_idx(k_page_offsets);
            }
        };

        auto V_mem_load = [&](auto v_lds_write_idx) {
            // V async load, symmetric to K_mem_load: with kFA4WG0LoadsV only WG0's 4
            // waves issue it; cooperative (=false) spreads it over all 8 waves.
            // v_block_idx stays uniform across all waves for loop / buffer bookkeeping.
            if(v_load_active)
            {
                if(cache_ptr_int32_overflow_possible)
                    async_load_tile_raw_long(v_lds_window_store(v_lds_write_idx), v_dram_window);
                else
                    async_load_tile_raw(v_lds_window_store(v_lds_write_idx), v_dram_window);
            }
            v_block_idx++;
            if(v_load_active && v_block_idx < num_iters_per_split)
            {
                refresh_v_offsets(v_block_idx, std::false_type{});
                if constexpr(kRebaseVSrd)
                    v_dram_window.rebase_buffer_base(v_pool_base + v_srd_base_offset);
                else
                    v_dram_window.update_page_idx(v_page_offsets);
            }
        };

        auto K_lds_load = [&](auto k_lds_read_idx) {
            kv_tile.k_tile = load_tile(k_lds_window_load(k_lds_read_idx));
        };

        auto V_lds_load = [&](auto v_lds_read_idx) {
            kv_tile.v_tile = load_tile_transpose(v_lds_window_load(v_lds_read_idx));
        };

        decltype(m) m_old;
        SMPLComputeDataType o_acc_scale; // rescale o_acc in fmha_alu1() & fmha_alu_D_upd()
        /// TODO: remove the sp_delta and use sp_compute directly
        // sp_delta follows sp: single slot for the kv128 tile, double otherwise.
        struct sp_delta_holder_t
        {
            decltype(sp(number<0>{}).sp_compute) d_;
            CK_TILE_DEVICE constexpr decltype(d_)& operator()(index_t) { return d_; }
        };
        std::conditional_t<kUseSingleSp,
                           sp_delta_holder_t,
                           statically_indexed_array<decltype(sp(number<0>{}).sp_compute), 2>>
            sp_delta;

        // Schraudolph exp2 approximation is only applied on the packed-shift,
        // non-masked path (the masked/causal path keeps the exact v_exp_f32, like
        // the ASM softmax). When active, fmha_alu0 folds the 2^23 scale + bias into
        // the shift FMA so fmha_alu1 finishes the exp with a single v_cvt_u32_f32.
        constexpr bool kUseExp2Approx =
            (UA_FA4_EXP2_APPROX != 0) && (UA_FA4_PACKED_SHIFT != 0) && !FmhaMask::IsMasking;

        auto fmha_alu0 = [&](auto sp_reg_idx) {
            m_old = m; // m{j-1}
            static_assert(m.thread_buf_.size() == 1,
                          "assuming that each thread holds 1 rowmax value");
            auto m_latest = block_tile_reduce<SMPLComputeDataType>(
                sp(sp_reg_idx).sp_compute, sequence<1>{}, f_max, m.thread_buf_[0]);
#if defined(__gfx950__)
            if constexpr(kWarpGemmM == 32)
            {
                int32x2_t swapped_regs =
                    __builtin_amdgcn_permlane32_swap(bit_cast<int32_t>(m_latest.thread_buf_[0]),
                                                     bit_cast<int32_t>(m_latest.thread_buf_[0]),
                                                     false,
                                                     false);
                m_latest.thread_buf_[0] = f_max(bit_cast<SMPLComputeDataType>(swapped_regs.x),
                                                bit_cast<SMPLComputeDataType>(swapped_regs.y));
            }
            else
            {
                block_tile_reduce_sync(m_latest, f_max, bool_constant<false>{});
            }
#else
            block_tile_reduce_sync(m_latest, f_max, bool_constant<false>{});
#endif
            m = m_latest;
#if CONDITIONAL_RESCALE
            if constexpr(kCondRescale)
            {
                // Decide — wave-uniformly — whether the true running max has
                // pulled more than τ ahead of the committed max. m / m_commit
                // are row-uniform after the cross-lane reduce, but the two
                // 32-lane row groups of the wave hold different rows, so OR the
                // per-lane predicate across the whole wave (ballot): if either
                // group needs a rescale, both commit. The other group then does
                // a near-no-op rescale, but the guard branch downstream stays
                // wave-uniform (ballot result in an SGPR → scalar s_cbranch).
                const bool nr_local =
                    (scale_s * (m.thread_buf_[0] - m_commit.thread_buf_[0])) >
                    CONDITIONAL_RESCALE_TAU;
                need_rescale                = (__builtin_amdgcn_ballot_w64(nr_local) != 0ull);
                m_commit_old.thread_buf_[0] = m_commit.thread_buf_[0];
                if(need_rescale)
                {
                    m_commit.thread_buf_[0] = m.thread_buf_[0];
                }
            }
#endif
            // Score-shift base: committed max for the prefill conditional path
            // (bounded by exp2(scale_s*(rowmax - m_commit)) ≤ exp2(τ) since we
            // commit whenever the gap exceeds τ), true running max otherwise.
#if CONDITIONAL_RESCALE
            auto& m_shift = kCondRescale ? m_commit : m;
#else
            auto& m_shift = m;
#endif

#if UA_FA4_PACKED_SHIFT
            // Packed score shift: each thread holds exactly one rowmax, so the FMA
            // addend (-scale_s * rowmax) is uniform across the thread's score
            // elements. Broadcast scale_s and the addend into both packed lanes and
            // emit v_pk_fma_f32 (2 f32/instr) over sp_compute.thread_buf_ pairs.
            // Bit-identical to the scalar fma_impl_vsv sweep below. The
            // one-rowmax-per-thread invariant is asserted on `m` above.
            static_assert(sp(sp_reg_idx).sp_compute.thread_buf_.size() % 2 == 0,
                          "packed shift needs an even score-register count");
            {
                // Schraudolph fold: bits = S*(scale_s*2^23) + (-scale_s*2^23*max +
                // bias) = 2^23*scale_s*(S-max) + bias, finished by v_cvt_u32_f32 in
                // fmha_alu1. Exact path: sp_delta = scale_s*(S-max).
                const float eff_scale =
                    kUseExp2Approx ? (scale_s * UA_EXP2_SCHRAUDOLPH_SCALE) : scale_s;
                const float addend =
                    kUseExp2Approx
                        ? (-eff_scale * m_shift.thread_buf_[0] + UA_EXP2_SCHRAUDOLPH_BIAS)
                        : (-scale_s * m_shift.thread_buf_[0]);
                const fp32x2_t scale_pair{eff_scale, eff_scale};
                const fp32x2_t addend_pair{addend, addend};
                static_for<0, sp(sp_reg_idx).sp_compute.thread_buf_.size(), 2>{}([&](auto idx) {
                    fp32x2_t in;
                    in.x        = sp(sp_reg_idx).sp_compute.thread_buf_[idx];
                    in.y        = sp(sp_reg_idx).sp_compute.thread_buf_[idx + 1];
                    auto out    = detail::pk_fma_f32(in, scale_pair, addend_pair);
                    sp_delta(sp_reg_idx).thread_buf_[idx]     = out.x;
                    sp_delta(sp_reg_idx).thread_buf_[idx + 1] = out.y;
                });
            }
#else
            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx        = make_tuple(idx0, idx1);
                    sp_delta(sp_reg_idx)(i_j_idx) = detail::fma_impl_vsv(
                        sp(sp_reg_idx).sp_compute(i_j_idx), scale_s, -scale_s * m_shift(i_j_idx));
                });
            });
#endif
            /// TODO: move some fmha_alu1() code here if necessary
        };

        auto fmha_alu1 = [&](auto sp_reg_idx) {
            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            if constexpr(kUseExp2Approx)
            {
                // fmha_alu0 already produced the Schraudolph bits; finish with a
                // single full-rate v_cvt_u32_f32 per element (no v_exp_f32).
                static_for<0, sp(sp_reg_idx).sp_compute.thread_buf_.size(), 1>{}([&](auto idx) {
                    sp(sp_reg_idx).sp_compute.thread_buf_[idx] =
                        detail::exp2_schraudolph_u32(sp_delta(sp_reg_idx).thread_buf_[idx]);
                });
            }
            else
            {
                sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        sp(sp_reg_idx).sp_compute(i_j_idx) =
                            ck_tile::exp2(sp_delta(sp_reg_idx)(i_j_idx));
                    });
                });
            }

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                sp(sp_reg_idx).sp_compute,
                sequence<1>{},
                f_sum,
                SMPLComputeDataType{0}); // rowsum(Pcompute{j})
            static_assert(rowsum_p.thread_buf_.size() == 1,
                          "assuming that each thread holds 1 rowsum value");
#if UA_FA4_PACKED_ROWSUM
            // Packed in-thread row-sum: accumulate pairs with v_pk_add_f32 into a
            // 2-wide partial, then one scalar combine. Overwrites the scalar
            // block_tile_reduce result above (its adds are dead -> DCE'd) before the
            // cross-lane reduce consumes thread_buf_[0].
            static_assert(sp(sp_reg_idx).sp_compute.thread_buf_.size() % 2 == 0,
                          "packed rowsum needs an even score-register count");
            {
                fp32x2_t acc{SMPLComputeDataType{0}, SMPLComputeDataType{0}};
                static_for<0, sp(sp_reg_idx).sp_compute.thread_buf_.size(), 2>{}([&](auto idx) {
                    fp32x2_t v;
                    v.x = sp(sp_reg_idx).sp_compute.thread_buf_[idx];
                    v.y = sp(sp_reg_idx).sp_compute.thread_buf_[idx + 1];
                    acc = detail::pk_add_f32(acc, v);
                });
                rowsum_p.thread_buf_[0] = acc.x + acc.y;
            }
#endif
#if defined(__gfx950__)
            if constexpr(kWarpGemmM == 32)
            {
                int32x2_t swapped_regs =
                    __builtin_amdgcn_permlane32_swap(bit_cast<int32_t>(rowsum_p.thread_buf_[0]),
                                                     bit_cast<int32_t>(rowsum_p.thread_buf_[0]),
                                                     false,
                                                     false);
                rowsum_p.thread_buf_[0] = f_sum(bit_cast<SMPLComputeDataType>(swapped_regs.x),
                                                bit_cast<SMPLComputeDataType>(swapped_regs.y));
            }
            else
            {
                block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
            }
#else
            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
#endif

            // l{j}
            /// Note: The compiler keeps moving the following instructions elsewhere because 'l'
            /// is first consumed later. To anchor them here, we rewrite the final addition in
            /// inline assembly to create a dependency, forcing the dependent instructions to
            /// be emitted at this point.
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CONDITIONAL_RESCALE
                // Denominator rescale uses the committed-max delta; == 1 (no-op
                // add) on the ~85-99% of tiles that don't commit. m_commit /
                // m_commit_old still hold this tile's values here (same
                // lifetime as m / m_old in the baseline).
                const auto tmp =
                    kCondRescale
                        ? ck_tile::exp2(scale_s * (m_commit_old[i_idx] - m_commit[i_idx]))
                        : ck_tile::exp2(scale_s * (m_old[i_idx] - m[i_idx]));
#else
                const auto tmp = ck_tile::exp2(scale_s * (m_old[i_idx] - m[i_idx]));
#endif
                l(i_idx) = detail::add_impl_vv(tmp * l[i_idx], rowsum_p[i_idx]);
            });

            // update partial o_acc [0, fmha_alu_D_reg_cnt)
#if CONDITIONAL_RESCALE
            // Skip the 6-VGPR partial rescale unless this tile committed. The
            // o_acc_scale + need_rescale read here are the values deferred from
            // the matching fmha_alu_D_upd (one pipeline stage back), exactly as
            // the baseline defers the o_acc_scale read. `!kCondRescale` folds
            // the guard away (unconditional rescale) on the decode path.
            if(!kCondRescale || need_rescale)
#endif
#if UA_FA4_PACKED_ALU1_RESCALE
            {
                static_assert(fmha_alu_D_reg_cnt % 2 == 0,
                              "packed alu1 rescale needs an even reg count");
                const fp32x2_t pk_scale{o_acc_scale, o_acc_scale};
                static_for<0, fmha_alu_D_reg_cnt, 2>{}([&](auto idx) {
                    fp32x2_t in;
                    in.x        = o_acc.thread_buf_[idx];
                    in.y        = o_acc.thread_buf_[idx + 1];
                    auto out    = detail::pk_mul_f32(in, pk_scale);
                    o_acc.thread_buf_[idx]     = out.x;
                    o_acc.thread_buf_[idx + 1] = out.y;
                });
            }
#else
            static_for<0, fmha_alu_D_reg_cnt, 1>{}([&](auto idx) {
                o_acc.thread_buf_[idx] = detail::mul_impl_vv(o_acc.thread_buf_[idx], o_acc_scale);
            });
#endif

            /// Note: The compiler keeps sinking the conversion instructions because the
            /// result 'p' is only consumed later. To anchor them here, we rewrite
            /// the cast_tile() call as inline assembly, forcing the conversions to be
            /// emitted at this point.
            static_assert(sp(sp_reg_idx).p.thread_buf_.size() % 2 == 0);
            if constexpr(std::is_same_v<PDataType, fp8_t>)
            {
                // FP8 P packing for the PV gemm.
                //
                // The CK reference path for fp32 -> fp8 is `cast_tile_pk_fp8_fp32`
                // in tile_elementwise.hpp, which CHAINS two `__builtin_amdgcn_cvt_pk_fp8_f32`
                // calls per 4-lane group: the second call uses the first call's
                // result as its `old` operand so the final uint32_t holds four
                // valid fp8 bytes packed into one register. Doing the conversion
                // pair-by-pair (one cvt_pk per 2 lanes, ignoring the upper 16
                // bits of the result) is *not* equivalent in practice -- the
                // builtin's `old` argument feeds the upper-bits passthrough and
                // when fed an uninitialised int the compiler is free to schedule
                // the cvt against junk that overlaps another live register,
                // which we observed as occasional whole-row output corruption
                // for FP8 prefill workloads.
                //
                // Use the chained 4-lanes-per-iteration pattern to match
                // `cast_tile_pk_fp8_fp32` byte-for-byte, then store as 4 fp8_t
                // bytes back into `p.thread_buf_`. We still anchor the work
                // inline (no `cast_tile(...)` indirection) so the conversions
                // stay at the end of `fmha_alu1` like the FP16/BF16 paths.
                static_assert(sp(sp_reg_idx).p.thread_buf_.size() % 4 == 0,
                              "fp8 P conversion expects packs of 4 fp32 lanes per "
                              "thread; widen the warp gemm M distribution if this "
                              "trips.");

                // ---------------------------------------------------------
                // FP8 P-tile QK-C -> PV-A re-layout.
                //
                // The CK UA pipeline relies on the QK-gemm's C output
                // aliasing byte-for-byte with the PV-gemm's A operand
                // through the `sp_compute` / `p` union. That alias is
                // only valid when the two warp gemms agree on per-
                // thread element ordering.
                //
                // For BF16/FP16 the PV gemm uses
                // `WGAttrNumAccess::Double` and the resulting layout
                // matches the QK-C `kCM0/kCMLane/kCM1` layout
                // byte-for-byte, so the alias just works.
                //
                // For FP8 the PV gemm is forced into `Single`
                // (load_tile_transpose's SubMinDim=8 constraint, see
                // GetPVBlockGemm in the policy) and the QK-C / PV-A
                // per-thread layouts diverge — naively reusing the
                // union feeds garbled data to the PV gemm.
                //
                // We have two re-layout strategies:
                //
                //   (A) Cross-lane in-register swap via
                //       `__builtin_amdgcn_ds_bpermute` between paired
                //       lanes (lane ^ 32). Cheap (one ds_bpermute_b32
                //       per PV K-iter, no LDS traffic, no barrier).
                //       Works for the 32x32 MFMA shapes (both K=16 and
                //       K=64): both have kAMLane=32 / kABKLane=2 and an
                //       identical 32x32 C-output distribution, so the
                //       QK-C -> PV-A relayout is the SAME paired-lane
                //       (bit-5) swap-of-half regardless of K. (The wider
                //       K=64 A-operand only changes kABKPerLane 8->32,
                //       i.e. the per-lane chunk COUNT, not the per-chunk
                //       swap pattern -- the 8-fp8 loop below just runs
                //       more iterations. Verified byte-identical to the
                //       narrow path on hw.)
                //
                //       NOTE: an earlier "cvt-only, layouts coincide"
                //       fast path for K=64 was WRONG -- QK-C holds one kv
                //       across many query rows while PV-A needs one query
                //       across many kv (a transpose), so skipping the
                //       swap silently corrupts P. The error was masked by
                //       near-uniform softmax (transposing a flat P barely
                //       moves the row-sum) and only surfaced as a few
                //       large-delta output lanes.
                //
                //   (B) Layout-agnostic LDS roundtrip via
                //       store_tile(QK-C dist) + s_barrier +
                //       load_tile(PV-A dist). Works for any MFMA
                //       shape, but adds ~1 LDS round-trip latency
                //       and a block-wide barrier per fmha_alu1 call.
                //       On 4-warp decode_m128 this measured ~2-3x
                //       worse end-to-end than (A).
                //
                // We pick (A) for the 32x32 tiers -- 32x32x16 (decode
                // m{32,64,128}) and 32x32x64 (wide-MMA prefill) -- and
                // (B) for the 16x16x32 m16 tiny-decode tier where (A)
                // doesn't apply. This keeps the previously-tuned 32x32x16
                // perf intact while enabling FP8 on the m16 tier.
                //
                // For strategy (A) the cvt and the cross-lane swap are
                // fused into a single 8-fp8-per-iter loop so that the
                // ds_bpermute_b32 latency overlaps with subsequent
                // cvt_pk_fp8_f32 calls (instead of running serially
                // after the whole cvt phase finishes).
                using PVWarpTile = typename UnifiedAttentionShape::Gemm1WarpTile;
                if constexpr(PVWarpTile::at(number<0>{}) == 32 &&
                             PVWarpTile::at(number<1>{}) == 32 &&
                             (PVWarpTile::at(number<2>{}) == 16 ||
                              PVWarpTile::at(number<2>{}) == 64))
                {
                    // ---- (A) Fused cvt + cross-lane swap (32x32x16 / 32x32x64). ----
                    //
                    // Per 8-fp8 K-chunk:
                    //   1. cvt 8 fp32 -> 2 packed uint32 (lo_pack = slot[0..3],
                    //      hi_pack = slot[4..7]) using the chained-`old` pattern
                    //      that matches `cast_tile_pk_fp8_fp32`.
                    //   2. ds_bpermute the "bad" pack to the paired lane (lane^32).
                    //   3. Write back both packs as 8 fp8 bytes; the "good" half
                    //      gets written first so its byte stores overlap with the
                    //      ds_bpermute latency.
                    //
                    // Slot decomposition (per fmha_alu1 doc above):
                    //   sub=0 | slot[0..3] | N=0..3   | K=0..3 OK
                    //   sub=0 | slot[4..7] | N=8..11  | K=4..7 BAD
                    //   sub=1 | slot[0..3] | N=4..7   | K=8..11 BAD
                    //   sub=1 | slot[4..7] | N=12..15 | K=12..15 OK
                    static_assert(sp(sp_reg_idx).p.thread_buf_.size() % 8 == 0,
                                  "FP8 32x32 (K=16/K=64) cross-lane permute "
                                  "expects PV per-thread buffer in chunks of 8 "
                                  "fp8 (one 32x32x16 warp-gemm K iteration worth "
                                  "of the swap-of-half pattern).");

                    // On gfx950 the paired-lane (l^32) swap is a single VALU
                    // op (v_permlane32_swap_b32), so the lane-id / ds_bpermute
                    // address machinery below is only needed for the
                    // ds_bpermute fallback on older arches (e.g. gfx942).
#if !defined(__gfx950__)
                    const int lane_id     = ck_tile::get_lane_id();
                    const int paired_addr = (lane_id ^ 32) << 2; // bytes
                    const bool is_sub_0   = (lane_id & 32) == 0;
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
                    int dummy_old;
                    static_for<0, sp(sp_reg_idx).p.thread_buf_.size(), 8>{}([&](auto k_base) {
                        auto& p  = sp(sp_reg_idx).p;
                        auto& sc = sp(sp_reg_idx).sp_compute;

                        const float a = p_compute_element_func(sc.thread_buf_[k_base + 0]);
                        const float b = p_compute_element_func(sc.thread_buf_[k_base + 1]);
                        const float c = p_compute_element_func(sc.thread_buf_[k_base + 2]);
                        const float d = p_compute_element_func(sc.thread_buf_[k_base + 3]);
                        const float e = p_compute_element_func(sc.thread_buf_[k_base + 4]);
                        const float f = p_compute_element_func(sc.thread_buf_[k_base + 5]);
                        const float g = p_compute_element_func(sc.thread_buf_[k_base + 6]);
                        const float h = p_compute_element_func(sc.thread_buf_[k_base + 7]);

                        const uint32_t lo_tmp =
                            __builtin_amdgcn_cvt_pk_fp8_f32(a, b, dummy_old, /*hi=*/false);
                        const uint32_t lo_pack =
                            __builtin_amdgcn_cvt_pk_fp8_f32(c, d, lo_tmp, /*hi=*/true);
                        const uint32_t hi_tmp =
                            __builtin_amdgcn_cvt_pk_fp8_f32(e, f, dummy_old, /*hi=*/false);
                        const uint32_t hi_pack =
                            __builtin_amdgcn_cvt_pk_fp8_f32(g, h, hi_tmp, /*hi=*/true);

#if defined(__gfx950__)
                        // gfx950: do the paired-lane (l^32) swap in a single
                        // VALU instruction instead of an LDS round-trip.
                        // v_permlane32_swap_b32 exchanges operand0's high half
                        // (lanes 32..63) with operand1's low half (lanes 0..31)
                        // and keeps the diagonal halves, so feeding
                        // (lo_pack, hi_pack) returns {out_lo, out_hi} directly
                        // for every lane -- the swap and the per-lane sub-block
                        // muxing the ds_bpermute path needs are both folded into
                        // the instruction. (Semantics verified on hw.)
                        const auto swapped =
                            __builtin_amdgcn_permlane32_swap(lo_pack, hi_pack,
                                                             /*fi=*/false,
                                                             /*bound_ctrl=*/false);
                        const uint32_t out_lo = swapped[0];
                        const uint32_t out_hi = swapped[1];
#else
                        // Issue ds_bpermute as early as possible so its LDS-DMA
                        // latency overlaps with the byte writes below (and with
                        // the next K-chunk's cvts after this iter unrolls).
                        const uint32_t own_bad = is_sub_0 ? hi_pack : lo_pack;
                        const uint32_t recv    = __builtin_amdgcn_ds_bpermute(
                            paired_addr, static_cast<int>(own_bad));

                        const uint32_t out_lo = is_sub_0 ? lo_pack : recv;
                        const uint32_t out_hi = is_sub_0 ? recv    : hi_pack;
#endif

                        p.thread_buf_[k_base + 0] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_lo >>  0) & 0xFFu));
                        p.thread_buf_[k_base + 1] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_lo >>  8) & 0xFFu));
                        p.thread_buf_[k_base + 2] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_lo >> 16) & 0xFFu));
                        p.thread_buf_[k_base + 3] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_lo >> 24) & 0xFFu));
                        p.thread_buf_[k_base + 4] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_hi >>  0) & 0xFFu));
                        p.thread_buf_[k_base + 5] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_hi >>  8) & 0xFFu));
                        p.thread_buf_[k_base + 6] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_hi >> 16) & 0xFFu));
                        p.thread_buf_[k_base + 7] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_hi >> 24) & 0xFFu));
                    });
#pragma clang diagnostic pop
                }
                else
                {
                    // ---- (B) LDS roundtrip (16x16x32 and any other
                    // future MFMA shape that doesn't fit the
                    // paired-lane swap pattern). ----
                    //
                    // The cvt phase is separated out here (vs fused into
                    // the swap as in branch (A)) because the relayout
                    // travels through LDS, not through `ds_bpermute`, so
                    // there's no swap-latency to hide.
                    //
                    //   1. cvt_pk_fp8_f32 chain into `sp(idx).p.thread_buf_`.
                    //   2. `p_qkc` is a static_distributed_tensor<fp8>
                    //      whose distribution metadata says "QK-C
                    //      layout". Its register bytes are populated
                    //      from `sp(idx).p.thread_buf_`, which is
                    //      exactly where the cvt_pk_fp8_f32 chain just
                    //      wrote the FP8 bytes (the union has them at
                    //      QK-C-layout register offsets).
                    //   3. `store_tile` writes `p_qkc` to LDS at
                    //      canonical (M, N) order.
                    //   4. Block-level barrier.
                    //   5. `load_tile` reads from the same LDS region
                    //      with the PV-A distribution.
                    //   6. Copy `p_pva.thread_buf_` back into
                    //      `sp(idx).p` so the gemm_1 call site reads
                    //      correctly-laid-out data with no further
                    //      changes.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
                    int dummy_old;
                    static_for<0, sp(sp_reg_idx).p.thread_buf_.size(), 4>{}([&](auto idx) {
                        const float a = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 0]);
                        const float b = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 1]);
                        const float c = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 2]);
                        const float d = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 3]);

                        const uint32_t lo =
                            __builtin_amdgcn_cvt_pk_fp8_f32(a, b, dummy_old, /*hi=*/false);
                        const uint32_t packed =
                            __builtin_amdgcn_cvt_pk_fp8_f32(c, d, lo, /*hi=*/true);
                        sp(sp_reg_idx).p.thread_buf_[idx + 0] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((packed >>  0) & 0xFFu));
                        sp(sp_reg_idx).p.thread_buf_[idx + 1] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((packed >>  8) & 0xFFu));
                        sp(sp_reg_idx).p.thread_buf_[idx + 2] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((packed >> 16) & 0xFFu));
                        sp(sp_reg_idx).p.thread_buf_[idx + 3] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((packed >> 24) & 0xFFu));
                    });
#pragma clang diagnostic pop

                    auto p_qkc = make_static_distributed_tensor<PDataType>(
                        sp(sp_reg_idx).sp_compute.get_tile_distribution());
                    static_assert(
                        decltype(p_qkc.thread_buf_)::size() ==
                            decltype(sp(sp_reg_idx).p.thread_buf_)::size(),
                        "QK-C and PV-A per-thread fp8 buffers must have the same "
                        "element count for the LDS roundtrip aliasing to be valid; "
                        "this should hold by construction since the union shares "
                        "register storage between sp_compute and p.");
                    static_for<0, decltype(p_qkc.thread_buf_)::size(), 1>{}(
                        [&](auto i) {
                            p_qkc.thread_buf_[i] = sp(sp_reg_idx).p.thread_buf_[i];
                        });

                    __builtin_amdgcn_s_barrier();
                    store_tile(p_lds_store_window_qkc, p_qkc);
                    __builtin_amdgcn_s_barrier();
                    auto p_pva = load_tile(p_lds_load_window_pva);
                    static_for<0, decltype(p_pva.thread_buf_)::size(), 1>{}(
                        [&](auto i) {
                            sp(sp_reg_idx).p.thread_buf_[i] = p_pva.thread_buf_[i];
                        });
                }
            }
            else
            {
                static_for<0, sp(sp_reg_idx).p.thread_buf_.size(), 2>{}([&](auto idx) {
                    float x = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx]);
                    float y = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 1]);
                    if constexpr(std::is_same_v<PDataType, fp16_t>)
                    {
                        auto casted                           = detail::cvt_pk_fp16_f32(x, y);
                        sp(sp_reg_idx).p.thread_buf_[idx]     = casted.x;
                        sp(sp_reg_idx).p.thread_buf_[idx + 1] = casted.y;
                    }
                    else
                    {
                        auto casted                           = cvt_pk_bf16_f32(x, y);
                        sp(sp_reg_idx).p.thread_buf_[idx]     = casted.x;
                        sp(sp_reg_idx).p.thread_buf_[idx + 1] = casted.y;
                    }
                });
            }

            /// Note: Place fmha_alu1() at the end of the phase. The surrounding inline assembly
            /// can interfere with the behavior of sched_group_barrier(), so ending the phase here
            /// avoids unintended reordering.
        };

        auto gemm = [&](auto sp_reg_idx, auto gemm_idx) {
            if constexpr(gemm_idx == 0)
            {
                clear_tile(sp(sp_reg_idx).sp_compute); // initialize C
                gemm_0(sp(sp_reg_idx).sp_compute,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kHeadDimPadded>{},
                                      sequence<kBlockM, k0_loops * kHeadDimPadded>{}),
                       get_slice_tile(kv_tile.k_tile,
                                      sequence<0, (k0_loops - 1) * kHeadDimPadded>{},
                                      sequence<kPageBlockSize, k0_loops * kHeadDimPadded>{}));
            }
            else
            {
                gemm_1(o_acc,
                       get_slice_tile(sp(sp_reg_idx).p,
                                      sequence<0, (k1_loops - 1) * kPageBlockSize>{},
                                      sequence<kBlockM, k1_loops * kPageBlockSize>{}),
                       get_slice_tile(kv_tile.v_tile,
                                      sequence<0, (k1_loops - 1) * kPageBlockSize>{},
                                      sequence<kHeadDimPadded, k1_loops * kPageBlockSize>{}));
            }
        };

        auto cl_calc = [&](auto sp_reg_idx, auto gemm_idx) {
#if UA_DYNAMIC_SETPRIO
            // Raise priority for the MFMA cluster so the computing warp group
            // outbids the co-resident memory-issuing group at the shared SIMD
            // issue port (HipKittens 8-wave ping-pong pattern).
            __builtin_amdgcn_s_setprio(1);
#endif
            if constexpr(gemm_idx == 0)
            {
                clear_tile(sp(sp_reg_idx).sp_compute); // initialize C
                gemm_0(sp(sp_reg_idx).sp_compute,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kHeadDimPadded>{},
                                      sequence<kBlockM, k0_loops * kHeadDimPadded>{}),
                       get_slice_tile(kv_tile.k_tile,
                                      sequence<0, (k0_loops - 1) * kHeadDimPadded>{},
                                      sequence<kPageBlockSize, k0_loops * kHeadDimPadded>{}));
            }
            else
            {
                gemm_1(o_acc,
                       get_slice_tile(sp(sp_reg_idx).p,
                                      sequence<0, (k1_loops - 1) * kPageBlockSize>{},
                                      sequence<kBlockM, k1_loops * kPageBlockSize>{}),
                       get_slice_tile(kv_tile.v_tile,
                                      sequence<0, (k1_loops - 1) * kPageBlockSize>{},
                                      sequence<kHeadDimPadded, k1_loops * kPageBlockSize>{}));
                fmha_alu0(number<1>{} - sp_reg_idx);
            }
#if UA_DYNAMIC_SETPRIO
            __builtin_amdgcn_s_setprio(0);
#endif
        };

        auto fmha_alu_D_upd = [&] {
#if CONDITIONAL_RESCALE
            // exp2(0) == 1.0 exactly on non-committing tiles; the guarded
            // multiplies below are then skipped entirely (scalar s_cbranch on
            // the wave-uniform need_rescale).
            o_acc_scale =
                kCondRescale
                    ? ck_tile::exp2(scale_s *
                                    (m_commit_old.thread_buf_[0] - m_commit.thread_buf_[0]))
                    : ck_tile::exp2(scale_s * (m_old.thread_buf_[0] - m.thread_buf_[0]));
#else
            o_acc_scale = ck_tile::exp2(scale_s * (m_old.thread_buf_[0] - m.thread_buf_[0]));
#endif

            static_assert((o_acc.thread_buf_.size() - fmha_alu_D_reg_cnt) % 2 == 0);

#if CONDITIONAL_RESCALE
            if(!kCondRescale || need_rescale)
            {
#endif
                fp32x2_t pk_o_acc_scale;
                pk_o_acc_scale.x = o_acc_scale;
                pk_o_acc_scale.y = o_acc_scale;

#if CK_TILE_DISABLE_PACKED_FP32
                static_assert(fmha_alu_D_reg_cnt + 2 <= o_acc.thread_buf_.size());
                static_for<fmha_alu_D_reg_cnt, fmha_alu_D_reg_cnt + 2, 1>{}(
                    [&](auto idx) { o_acc.thread_buf_[idx] *= o_acc_scale; });
#endif

                constexpr auto issued_D_reg_cnt =
#if CK_TILE_DISABLE_PACKED_FP32
                    fmha_alu_D_reg_cnt + 2
#else
                    fmha_alu_D_reg_cnt
#endif
                    ;
                /// NOTICE: Use inline asm v_pk_mul_f32 to reduce latency. The fmha_alu_D_upd() call
                /// should be placed at the end of a phase.
                // update partial o_acc after [issued_D_reg_cnt]
                static_for<issued_D_reg_cnt, o_acc.thread_buf_.size(), 2>{}([&](auto idx) {
                    fp32x2_t input;
                    input.x = o_acc.thread_buf_[idx];
                    input.y = o_acc.thread_buf_[idx + 1];

                    auto output = detail::pk_mul_f32(input, pk_o_acc_scale);

                    o_acc.thread_buf_[idx]     = output.x;
                    o_acc.thread_buf_[idx + 1] = output.y;
                });
#if CONDITIONAL_RESCALE
            }
#endif
        };

        // Resolve kBlockQ at runtime when the caller plumbs in
        // num_queries_per_kv (=> kBlockQ = kBlockM / num_qpkv). Fall back to
        // the static `kBlockQ` from `UnifiedAttentionShape` when the caller
        // passes 0 (back-compat). Stored once, reused per K-tile mask check.
        const index_t kBlockQ_dyn =
            (num_queries_per_kv > 0) ? (kBlockM / num_queries_per_kv) : kBlockQ;

        auto fmha_mask = [&](auto sp_reg_idx) {
            if constexpr(FmhaMask::IsMasking)
            {
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           i_total_loops * kPageBlockSize,
                                                           kBlockQ_dyn,
                                                           static_cast<index_t>(kPageBlockSize));
                if(need_perpixel_check)
                {
                    set_tile_if(sp(sp_reg_idx).sp_compute,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&](auto tile_idx) {
                                    const auto row =
                                        q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                    const auto col =
                                        i_total_loops * kPageBlockSize + tile_idx.at(number<1>{});
                                    return mask.IsOutOfBound(row, col);
                                });
                }
            }
        };

        // FA4 mask variant: the matrix‖softmax split decouples "which KV tile's
        // scores we are masking" from the loop counter `i_total_loops` (the FA4
        // pre-stage pre-increments it, and the prime / deferred-softmax shift
        // the softmax of tile k to a point where i_total_loops != k). Passing
        // the absolute KV-tile index explicitly keeps the masked column window
        // aligned with the tile actually held in `sp(sp_reg_idx)`. For large
        // sk the early tiles are fully in-bounds (IsEdgeTile false) so the bug
        // was invisible; it only bit the causal diagonal (small sk) tiles.
        [[maybe_unused]] auto fmha_mask_at = [&](auto sp_reg_idx, index_t kv_tile_idx) {
            if constexpr(FmhaMask::IsMasking)
            {
                const index_t col_base   = kv_tile_idx * kPageBlockSize;
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           col_base,
                                                           kBlockQ_dyn,
                                                           static_cast<index_t>(kPageBlockSize));
                if(need_perpixel_check)
                {
                    set_tile_if(sp(sp_reg_idx).sp_compute,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&](auto tile_idx) {
                                    const auto row =
                                        q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                    const auto col = col_base + tile_idx.at(number<1>{});
                                    return mask.IsOutOfBound(row, col);
                                });
                }
            }
        };

        auto cl_load = [&](auto load_type, auto mem_wr_idx, auto lds_rd_idx) {
            if constexpr(load_type == 0)
            {
                V_mem_load(mem_wr_idx);
                K_lds_load(lds_rd_idx);
            }
            else
            {
                K_mem_load(mem_wr_idx);
                V_lds_load(lds_rd_idx);
            }
        };

        // ---------------------------------------------------------------
        // FA4 matrix‖softmax overlap core loop.
        //
        // Both warp groups run the SAME deferred-PV software pipeline, cut into
        // two barrier-delimited phases. Per iteration pi (sp slot the NEW QK
        // lands in is `1-pi`, matching the existing p01/p23 convention; the
        // tile whose softmax finishes / is PV'd is slot `pi`):
        //
        //   MATRIX(pi)  : PV(sp(pi))   = o_acc += P(pi) @ V(k-1)   (matrix)
        //                 QK(sp(1-pi)) = Q @ K(k)                  (matrix)
        //   SOFTMAX(pi) : mask, alu0, D_upd, alu1  on slot (1-pi)  (VALU/MUFU)
        //                 → produces P(1-pi) for the NEXT MATRIX's PV.
        //
        // NOTE on numerics: unlike the existing pipeline — which SPLITS the
        // o_acc rescale (6 VGPRs in fmha_alu1, the rest in fmha_alu_D_upd) and
        // defers the alu1 partial "one pipeline stage back" for latency — the
        // FA4 SOFTMAX phase co-locates alu0→D_upd→alu1 on a single slot, so the
        // whole o_acc is rescaled by THIS tile's o_acc_scale in one place. That
        // is the textbook online-softmax rescale: numerically equivalent, but
        // NOT bit-identical to the staged baseline. Keeping D_upd at the END of
        // SOFTMAX is what leaves MATRIX pure-matrix (no MFMA-waits-on-rescale
        // hazard, no cross-warp VALU contention).
        //
        // The two groups are primed one phase apart (see dispatch): cl_p == 0
        // runs MATRIX-then-SOFTMAX each slot, cl_p == 1 runs SOFTMAX-then-
        // MATRIX, so at every block barrier one group's matrix-pipe work hides
        // its co-resident partner's VALU/MUFU work. Only the MATRIX-phase group
        // touches K/V, so the shared double buffer has no cross-group conflict.
        //
        // K/V are prefetched a tile ahead at the block barrier, where all 8
        // warps are converged, so the block-cooperative async load covers the
        // full tile (a single 4-warp group can only load its own half-slice).
        //
        // Absolute KV-tile index of the NEXT tile to be softmaxed/masked in the
        // FA4 path. Incremented once per softmax (in strict tile order across
        // prime → loop → epilogue), so it always names the tile whose scores
        // sit in the slot being masked — independent of `i_total_loops` (which
        // FA4 uses only for loop control + prefetch bookkeeping). Starts at
        // num_blocks_start to honour the split-KV column offset, matching the
        // baseline's i_total_loops*kPageBlockSize convention.
        [[maybe_unused]] index_t fa4_sm_tile = num_blocks_start;

        // Gate the prefetch-in-softmax experiment to the 2-byte path: fp8 keeps
        // the matrix-phase prefetch it was tuned against; bf16/fp16 move the
        // async DRAM-load *issue* into the softmax phase (residency still
        // enforced by the next MATRIX's vmcnt drain + barrier — see macro note).
        constexpr bool kPrefetchInSoftmax =
            (UA_FA4_PREFETCH_IN_SOFTMAX != 0) && !std::is_same_v<KDataType, fp8_t>;

        [[maybe_unused]] auto core_loop_fa4 = [&](auto cl_p) {
            auto gemm0 = number<0>{};
            auto gemm1 = number<1>{};

            // MATRIX phase: deferred PV(k-1) then QK(k). Pure matrix pipe.
            // Consumes V(pi) / K(1-pi) resident in LDS; kv_tile holds v_tile for
            // the PV and (separately) k_tile for the QK. Both LDS reads (V then K)
            // live in this phase: V up front (overlaps the lgkmcnt drain), K
            // issued between the PV and QK MFMAs so its read overlaps the PV MFMA.
            auto fa4_matrix = [&](auto pi) {
                auto pv_sp = pi;               // PV source: P(pi) from prev SOFTMAX
                auto qk_sp = number<1>{} - pi; // QK target slot
                auto k_rd  = number<1>{} - pi;

                // V LDS read lives HERE in the MATRIX phase. PV(pi) consumes V
                // buf pi; that buffer was filled + published (drain-before-
                // barrier) by WG0 in a previous slot, so the slot barrier we just
                // crossed already guarantees residency. Issued first so its LDS
                // latency overlaps the lgkmcnt<0> below.
                V_lds_load(pi);
                s_waitcnt_lgkmcnt<0>(); // wait the V LDS read just issued
                gemm(pv_sp, gemm1);     // o_acc += P(pi) @ V(k-1)
                // K read into its OWN registers (k_tile no longer aliases v_tile),
                // so this ds_read executes on the LSU *during* the PV MFMA above
                // rather than waiting for it to retire; the sched_barriers pin it
                // here. K is now single-warp-group loaded (kFA4WG1LoadsK) so it is
                // resident at the slot-A barrier, but issuing the read AFTER the
                // PV gemm call (overlapping the in-flight MFMA) schedules strictly
                // better than hoisting it ahead of PV — measured ~3-4% faster.
                __builtin_amdgcn_sched_barrier(0);
                K_lds_load(k_rd); // overlaps the PV MFMA (latency hidden)
                __builtin_amdgcn_sched_barrier(0);
                s_waitcnt_lgkmcnt<0>();
                gemm(qk_sp, gemm0); // sp(1-pi).sp_compute = Q @ K(k)
            };

            // SOFTMAX phase on the just-QK'd slot (1-pi): mask, rowmax+shift
            // (alu0), accumulator rescale (D_upd — the O*corr tail), then
            // exp+rowsum+P-cvt (alu1), which produces P(1-pi) for the next
            // MATRIX phase's PV.
            auto fa4_softmax = [&](auto pi) {
                auto sm_sp = number<1>{} - pi;
                fmha_mask_at(sm_sp, fa4_sm_tile++);
                fmha_alu0(sm_sp);
                fmha_alu_D_upd();
                fmha_alu1(sm_sp);
#if UA_FA4_PIN_PACK_IN_SOFTMAX
                // Pin the fmha_alu1 P-pack (cvt_pk_fp8) inside this SOFTMAX
                // region: the scheduler may not sink it across this fence into
                // the following MATRIX slot, where it would contend with the
                // co-resident group's softmax VALU on the SIMD issue port.
                __builtin_amdgcn_sched_barrier(0);
#endif
            };

            // One KV tile == one MATRIX + one SOFTMAX phase, separated by two
            // block barriers. The prefetch for tile k+1 is issued right after
            // the FIRST barrier in BOTH branches, where all 8 warps are
            // converged, so the block-cooperative async load covers the full
            // tile (a 4-warp group only owns its half-slice). Buffer index ==
            // sp slot; prefetch targets the buffer the next iteration reads
            // (K→buf[pi], V→buf[1-pi]; the opposite buffers are being read this
            // iteration, so the double buffer never aliases).
            auto iteration = [&](auto pi) {
                bool result = true;
                auto K_pf = pi;               // next-tile K buffer (WG1 fills it)
                auto V_pf = number<1>{} - pi; // next-tile V buffer (WG0 fills it)

                // Load roles are warp-group-specialized: WG0 issues the V async
                // load, WG1 issues the K async load (the other group's call is a
                // no-op that only advances bookkeeping). Both groups READ both
                // tiles from the shared LDS double buffer.
                auto prefetch = [&] {
                    if(i_total_loops + 1 < num_total_loop)
                        K_mem_load(K_pf); // real on WG1, no-op on WG0
                    V_mem_load(V_pf);     // real on WG0, no-op on WG1
                };

                auto barrier = [] {
                    __builtin_amdgcn_sched_barrier(0); // pin: nothing crosses the
                    __builtin_amdgcn_s_barrier();      // block barrier (keeps the
                    __builtin_amdgcn_sched_barrier(0); // cooperative load converged)
                };

                if constexpr(cl_p == 0)
                {
                    // ================= WG0 : MATRIX(pi) then SOFTMAX(pi) ========
                    // WG0 is the V loader. Each slot drains its own outstanding V
                    // loads (vmcnt<0>) BEFORE the barrier, so the barrier publishes
                    // all 4 WG0 waves' cooperative writes to the readers (itself +
                    // WG1) that cross it next.

                    // ---- slot A: MATRIX(pi) ‖ WG1 SOFTMAX ----
                    ASM_MARKER("fa4 MATRIX Wave0-3");
                    s_waitcnt_vmcnt<0>(); // V for THIS matrix has arrived -> publish
                    barrier();
                    if constexpr(!kPrefetchInSoftmax)
                        prefetch();       // issue K(pi)+V(1-pi) for tile k+1
                    fa4_matrix(pi);       // V_lds_load(pi); PV; K_lds_load(1-pi); QK

                    // ---- slot B: SOFTMAX(pi) ‖ WG1 MATRIX ----
                    // No VMEM drain here: slot A's vmcnt<0> already published every
                    // V load, so the ONLY load outstanding now is the slot-A next-
                    // tile prefetch, which no reader touches for ~2 phases. Draining
                    // it here just landed it a phase early. The barrier still stays:
                    // it's the phase sync and publishes this phase's LDS (sp/P tiles)
                    // via the lgkm side, not VMEM.
                    ASM_MARKER("fa4 SOFTMAX Wave0-3");
                    barrier();
                    if constexpr(kPrefetchInSoftmax)
                        prefetch();       // bf16/fp16: kick the next-tile load here
                    fa4_softmax(pi);

                    if(num_total_loop <= ++i_total_loops)
                        result = false;
                }
                else
                {
                    // ================= WG1 : SOFTMAX then MATRIX(pi) ============
                    // WG1 is the K loader, primed one phase ahead of WG0.

                    // ---- slot A: SOFTMAX ‖ WG0 MATRIX ----
                    ASM_MARKER("fa4 SOFTMAX Wave4-7");
                    s_waitcnt_vmcnt<0>(); // K for WG0's matrix has arrived -> publish
                    barrier();
                    if constexpr(kPrefetchInSoftmax)
                        prefetch();       // bf16/fp16: kick the next-tile load here
                    fa4_softmax(number<1>{} - pi);

                    // ---- slot B: MATRIX(pi) ‖ WG0 SOFTMAX ----
                    // Prefetch tile k+1 from WG1's MATRIX slot (not its SOFTMAX
                    // slot above): the async K/V DRAM load issues while the partner
                    // group (WG0) runs SOFTMAX, so the load-issue overhead overlaps
                    // that softmax. Trade-off: the load lands one phase later than
                    // WG0's, shrinking the latency-hide window -- watch vmwait.
                    ASM_MARKER("fa4 MATRIX Wave4-7");
                    barrier();
                    if constexpr(!kPrefetchInSoftmax)
                        prefetch();       // issue K(pi)+V(1-pi) for tile k+1
                    fa4_matrix(pi);       // V_lds_load(pi); PV; K_lds_load(1-pi); QK

                    if(num_total_loop <= ++i_total_loops)
                        result = false;
                }
                return result;
            };
            return iteration(number<0>{}) && iteration(number<1>{});
        };

        // FA4 deferred-PV epilogue: the final SOFTMAX produced P for a tile
        // whose PV has not yet been folded into o_acc. Run that last PV here.
        // (alu1 already ran in the SOFTMAX phase, so unlike fmha_post_process
        // we do NOT re-run it — just the V load + PV gemm.)
        [[maybe_unused]] auto fa4_post_process = [&](auto last_pv_sp, auto last_v_buf) {
            s_waitcnt_vmcnt<0>();
            __builtin_amdgcn_s_barrier();
            V_lds_load(last_v_buf);
            s_waitcnt_lgkmcnt<0>();
            gemm(last_pv_sp, /*gemm_idx=*/number<1>{});
        };

        auto fmha_post_process = [&](auto d) {
            auto ps_pi        = number<1>{} - d;
            auto V_lds_rd_idx = ps_pi;

            // Wait for the last V tile's async load to complete before reading from LDS.
            // The main loop's final iteration never prefetches K (i_total_loops+1 ==
            // num_total_loop), so only V loads are outstanding here.  The original
            // s_waitcnt_vmcnt<K_mem_su_ld_insts> was a no-op when V_su_ld_insts ==
            // K_su_ld_insts (e.g. both 2 for kPageBlockSize=32), causing a race where
            // V_lds_load read stale LDS before the async V load finished.
            s_waitcnt_vmcnt<0>();
            __builtin_amdgcn_s_barrier();

            V_lds_load(V_lds_rd_idx);
            fmha_alu1(ps_pi);

            s_waitcnt_lgkmcnt<0>();

            auto xdl_SP_p23_reg_idx = ps_pi;
            gemm(xdl_SP_p23_reg_idx, /*gemm_idx=*/number<1>{});
        };

        // pre-stage
        {
            ASM_MARKER("before pre-stage");
            // (1) load K0 to LDS & VGPR
            K_mem_load(number<0>{}); // mem_K0

            s_waitcnt_vmcnt<0>();
            __builtin_amdgcn_s_barrier();

            K_lds_load(number<0>{}); // lds_K0

            s_waitcnt_lgkmcnt<0>();
            __builtin_amdgcn_s_barrier();

            // (2) prefetch K1 and V0 to LDS in parallel with GEMM0
            if(1 < num_total_loop)
            {
                K_mem_load(number<1>{}); // mem_K1
            }
            V_mem_load(number<0>{}); // mem_V0

            // (3) mfma (Q*K0) + softmax
            gemm(number<0>{}, /*gemm_idx=*/number<0>{});

            // FA4 prefills sp(0) with the raw QK(0) only. The softmax of tile 0
            // is done by each warp group itself: the softmax-first group folds
            // it into its first loop iteration; the matrix-first group runs it
            // as a one-shot prime in the dispatch below. The K2 prefetch is
            // also skipped — the FA4 loop prefetches exactly one tile ahead, so
            // the first iteration must issue K2 itself (issuing it here would
            // leave the loop one K tile too far ahead and clobber K2 with K3).
            if constexpr(kFA4)
            {
                ++i_total_loops;
                if(num_total_loop <= i_total_loops)
                {
                    goto label_main_loops_exit;
                }
                ASM_MARKER("end pre-stage (FA4)");
            }
            else
            {
                fmha_mask(number<0>{});
                /// TODO: find better way to map fmha_alu(0,96) call
                fmha_alu0(number<0>{});
                fmha_alu_D_upd();

                ++i_total_loops;
                if(num_total_loop <= i_total_loops)
                {
                    goto label_main_loops_exit;
                }

                if(2 < num_total_loop)
                {
                    K_mem_load(number<0>{}); // mem_K2

                    s_waitcnt_vmcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    __builtin_amdgcn_s_barrier();
                }

                ASM_MARKER("end pre-stage");
            }
        }

        if(1 < num_total_loop)
        {
            if constexpr(NumWarpGroups == 1)
            {
                // --- Single warp group: serial pipeline with async prefetch ---
                // After pre-stage:
                //   sp(0) has QK for block 0 (alu0 + alu_D_upd done, alu1 NOT done)
                //   V0 loading to LDS (V buf 0)
                //   K1 in LDS (K buf 1) if num_total_loop >= 2
                //   K2 loading to LDS (K buf 0) if num_total_loop >= 3

                // Step 1: consume V0, K1 -> produce PV(0), QK(1)
                s_waitcnt_vmcnt<0>();
                __builtin_amdgcn_s_barrier();

                V_mem_load(number<1>{}); // prefetch V1 -> buf 1 (overlaps with compute)

                V_lds_load(number<0>{}); // V0 from LDS -> kv_tile.v_tile
                s_waitcnt_lgkmcnt<0>();
                fmha_alu1(number<0>{}); // finalize sp(0) -> P(0)
                gemm(number<0>{}, /*gemm_idx=*/number<1>{}); // PV: P(0)*V0

                K_lds_load(number<1>{}); // K1 from LDS -> kv_tile.k_tile
                s_waitcnt_lgkmcnt<0>();

                gemm(number<1>{}, /*gemm_idx=*/number<0>{}); // QK: Q*K1 -> sp(1)
                fmha_mask(number<1>{});
                fmha_alu0(number<1>{});
                fmha_alu_D_upd();
                i_total_loops++;

                while(i_total_loops < num_total_loop)
                {
                    // Even step: V from buf 1, K from buf 0, QK -> sp(0)
                    // kv_tile is a union: must finish PV GEMM (v_tile) before K load
                    s_waitcnt_vmcnt<0>();
                    __builtin_amdgcn_s_barrier();

                    // Prefetch next iteration's K/V (overlaps with all compute below)
                    // K/V use separate LDS regions so no conflict with current reads
                    if(i_total_loops + 1 < num_total_loop)
                        K_mem_load(number<1>{}); // next K -> K buf 1
                    V_mem_load(number<0>{}); // next V -> V buf 0

                    V_lds_load(number<1>{}); // V from V buf 1 -> kv_tile.v_tile
                    s_waitcnt_lgkmcnt<0>();
                    fmha_alu1(number<1>{}); // finalize sp(1) -> P(1)
                    gemm(number<1>{}, /*gemm_idx=*/number<1>{}); // PV: P(1)*V

                    K_lds_load(number<0>{}); // K from K buf 0 -> kv_tile.k_tile
                    s_waitcnt_lgkmcnt<0>();

                    gemm(number<0>{}, /*gemm_idx=*/number<0>{}); // QK -> sp(0)
                    fmha_mask(number<0>{});
                    fmha_alu0(number<0>{});
                    fmha_alu_D_upd();
                    i_total_loops++;

                    if(i_total_loops >= num_total_loop)
                        break;

                    // Odd step: V from buf 0, K from buf 1, QK -> sp(1)
                    s_waitcnt_vmcnt<0>();
                    __builtin_amdgcn_s_barrier();

                    // Prefetch next iteration's K/V
                    if(i_total_loops + 1 < num_total_loop)
                        K_mem_load(number<0>{}); // next K -> K buf 0
                    V_mem_load(number<1>{}); // next V -> V buf 1

                    V_lds_load(number<0>{}); // V from V buf 0 -> kv_tile.v_tile
                    s_waitcnt_lgkmcnt<0>();
                    fmha_alu1(number<0>{}); // finalize sp(0) -> P(0)
                    gemm(number<0>{}, /*gemm_idx=*/number<1>{}); // PV: P(0)*V

                    K_lds_load(number<1>{}); // K from K buf 1 -> kv_tile.k_tile
                    s_waitcnt_lgkmcnt<0>();

                    gemm(number<1>{}, /*gemm_idx=*/number<0>{}); // QK -> sp(1)
                    fmha_mask(number<1>{});
                    fmha_alu0(number<1>{});
                    fmha_alu_D_upd();
                    i_total_loops++;
                }
            }
            else
            {
                // --- Two warp groups: FA4 matrix‖softmax overlap ---
                // After the FA4 pre-stage sp(0) holds the raw QK(0). WG0 is the
                // matrix-first group: it primes tile-0's softmax once (barrier-
                // free: bf16/fp16 register cast, or FP8 strategy-A within-wave
                // permute) so its first MATRIX has P(0), then runs MATRIX-then-
                // SOFTMAX each slot. WG1 is softmax-first: it folds tile-0's
                // softmax into its first iteration and runs SOFTMAX-then-MATRIX,
                // so the two groups sit one phase apart and overlap on every
                // SIMD. (kFA4 is statically guaranteed true here — see the
                // static_assert at the kFA4 definition.)
                if(warp_group_id == 0)
                {
                    __builtin_amdgcn_s_setprio(0);
                    fmha_mask_at(number<0>{}, fa4_sm_tile++); // tile num_blocks_start
                    fmha_alu0(number<0>{});
                    fmha_alu_D_upd();
                    fmha_alu1(number<0>{}); // sp(0).p = P(0)
                    // Prime v_tile for the first MATRIX(0): V buf 0 was loaded by
                    // WG0 in the pre-stage, so its own vmcnt<0> proves residency.
                    // (Stage B reads each subsequent tile's V in the prior SOFTMAX.)
                    s_waitcnt_vmcnt<0>();
                    V_lds_load(number<0>{});
                    while(core_loop_fa4(number<0>{}))
                        ;
                }
                if(warp_group_id != 0)
                {
                    __builtin_amdgcn_s_setprio(0);
                    while(core_loop_fa4(number<1>{}))
                        ;
                }
            }
        }
    label_main_loops_exit:
        // The post-process call finalizes whichever SP register was left in
        // an "alu0-done, alu1-pending" state at the end of the main loop.
        // Which one that is depends on the parity of the *number of
        // iterations performed* (= num_total_loop - num_blocks_start), not
        // num_total_loop itself. For the non-split path num_blocks_start
        // is always 0 so the two parities coincide; the split-KV path with
        // num_blocks_start > 0 needs the corrected expression below.
        const index_t num_iters = num_total_loop - num_blocks_start;
        // FA4 drain (NumWarpGroups==2) vs baseline post-process (the
        // single-warp-group serial decode path, where kFA4 is false).
        if constexpr(kFA4)
        {
            // Deferred-PV drain. The pending sp slot has the same parity as the
            // baseline post-process slot (ps_pi == 1 - num_iters%2), and the
            // last tile's V sits in the buffer with that same index. WG0
            // (matrix-first) already softmaxed this slot inside the loop, so
            // only its deferred PV remains. WG1 (softmax-first) — and the
            // degenerate single-tile case, where the pre-stage jumped straight
            // here without priming or looping — still owes the full softmax of
            // that slot before the PV. Both branches issue exactly one
            // s_barrier (inside fa4_post_process; the softmax tail is
            // barrier-free — bf16/fp16 register cast or FP8 strategy-A
            // within-wave permute) so the two warp groups stay in lockstep.
            auto fa4_epi = [&](auto slot) {
                // WG1 always owes the final tile's softmax (softmax-first group
                // defers it one phase). WG0 (matrix-first) normally softmaxed it
                // inside the loop — EXCEPT in the degenerate num_iters==1 case
                // where the FA4 pre-stage jumped straight here without priming
                // or looping, so NEITHER group softmaxed it. That predicate is
                // num_iters==1, NOT num_total_loop==1: under split-KV a trailing
                // 1-tile split has num_total_loop = num_blocks_start+1 > 1 while
                // num_iters==1, and the old condition wrongly let WG0 PV an
                // un-softmaxed slot (garbage) for that split.
                if(warp_group_id != 0 || num_iters == 1)
                {
                    fmha_mask_at(slot, fa4_sm_tile++); // last tile (num_total_loop-1)
                    fmha_alu0(slot);
                    fmha_alu_D_upd();
                    fmha_alu1(slot);
                }
                fa4_post_process(slot, slot);
            };
            if(num_iters % 2)
                fa4_epi(number<0>{});
            if(!(num_iters % 2))
                fa4_epi(number<1>{});
        }
        else
        {
            if(num_iters % 2)
            {
                fmha_post_process(number<1>{});
            }
            if(!(num_iters % 2))
            {
                fmha_post_process(number<0>{});
            }
        }

        // finally, O — normalize by l
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            // Fuse the V FP8 descale into the per-row normalisation so the
            // post-loop pass touches o_acc only once. v_descale is host-set
            // to 1.0f for non-FP8 dtypes so this stays a free no-op there.
            // Masked rows that saw no valid keys keep their zeros (the
            // l == 0 short-circuit below).
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : v_descale / l[i_idx];
                }
                else
                    return v_descale / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        // Build the log-sum-exp side-output (natural-log domain) for the
        // split-KV combine kernel. For non-split callers this is ignored.
        //
        // Note `m` here is the *unscaled* rowmax of the raw qk dot products
        // (the pipeline computes `m = block_tile_reduce(sp_compute, max)`
        // before applying `scale_s`). Likewise `l = sum exp2(scale_s*(s-m))`
        // is the natural-domain softmax denominator (since `scale_s` carries
        // a baked-in log2(e), `exp2(scale_s*x) == exp(scale*x)`). Combined,
        //   LSE = log(sum exp(scale * s_k))
        //       = scale * m + log(l)
        //       = scale_s/log2(e) * m + log(l).
        // The combine kernel re-weights partials with exp(lse - lse_max).
        const auto scale_natlog =
            scale_s / static_cast<SMPLComputeDataType>(C_LOG2E);
        auto lse = make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
#if CONDITIONAL_RESCALE
        // o_acc/l are carried in the m_commit frame, so the denominator is
        // l = sum exp2(scale_s*(s - m_commit)). LSE = scale*m_commit + log(l)
        // is then exact (the frame cancels in o = o_acc/l regardless; only the
        // side-output LSE needs the matching base). m_commit ≤ true max but
        // that is precisely the frame l was summed in. Decode keeps m.
        sweep_tile_span(o_spans[number<0>{}],
                        [&, m_ = (kCondRescale ? m_commit : m), l_ = l](auto idx0) {
#else
        sweep_tile_span(o_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
#endif
            constexpr auto i_idx = make_tuple(idx0);
            if constexpr(FmhaMask::IsMasking)
            {
                lse(i_idx) =
                    (l_[i_idx] == 0.f)
                        ? -ck_tile::numeric<SMPLComputeDataType>::infinity()
                        : scale_natlog * m_[i_idx] + ck_tile::log(l_[i_idx]);
            }
            else
            {
                lse(i_idx) = scale_natlog * m_[i_idx] + ck_tile::log(l_[i_idx]);
            }
        });

        return ck_tile::make_tuple(o_acc, lse);
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(
        const QDramBlockWindowTmp& q_dram_block_window_tmp, // kBlockM * kHeadDimPadded tile
        const KDramBlockWindowTmp& k_dram_block_window_tmp, // kPageBlockSize * kHeadDimPadded tile
        const VDramBlockWindowTmp& v_dram_block_window_tmp, // kHeadDimPadded * kPageBlockSize tile
        const index_t num_blocks,
        const index_t num_blocks_start,
        const void* block_tables_ptr,
        index_t block_table_offset,
        const index_t page_size, // PageSize in tokens (cache rows per page)
        FmhaMask mask,
        float scale_s,
        void* smem_ptr,
        long_index_t k_row_stride        = 0,
        long_index_t v_row_stride        = 0,
        // Forwards to the full-args operator() so callers can plumb in a
        // runtime kBlockQ. See the documentation on that overload.
        const index_t num_queries_per_kv = 0,
        // See the doc on the full-args operator().
        const bool cache_ptr_int32_overflow_possible = false,
        // See the doc on the full-args operator(). Defaults to 1.0f so
        // non-FP8 callers see no behavior change.
        const float v_descale = 1.0f) const
    {
        using namespace ck_tile;

        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          num_blocks,
                          num_blocks_start,
                          block_tables_ptr,
                          block_table_offset,
                          page_size,
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          scale_s,
                          smem_ptr,
                          k_row_stride,
                          v_row_stride,
                          num_queries_per_kv,
                          cache_ptr_int32_overflow_possible,
                          v_descale);
    }
};

} // namespace ck_tile
