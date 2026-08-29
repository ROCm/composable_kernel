// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-only helpers for the FMHA BWD dQ/dK/dV kernel family.
//
// HOST ONLY: this header must NOT be included from device code (.hip files).
// Device code should include <rocm_ck/ops/fmha_bwd/dqdkdv_dev.hpp>.
//
// Compilation boundary:
//   _spec.hpp -- consteval factory + slot constants (both passes)
//   _api.hpp (this) -- host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp -- CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#error "dqdkdv_api.hpp is host-only." \
       " Device code should include <rocm_ck/ops/fmha_bwd/dqdkdv_dev.hpp>."
#endif

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

#include <rocm_ck/args.hpp>
#include <rocm_ck/grid_dim.hpp>

#ifndef NDEBUG
#include <cstdio>
#include <cstdlib>
#endif

namespace rocm_ck {

// ---------------------------------------------------------------------------
// Grid calculation
// ---------------------------------------------------------------------------

/// Compute the launch grid for dQ/dK/dV.
/// Matches CK Tile's FmhaBwdDQDKDVSpec::GridSize():
///   GridDim(ceil(seqlen_k / kN0), nhead, batch).
/// block_n0 comes from FmhaBwdDQDKDVSpec::block_n0 (kN0).
/// Precondition: block_n0 > 0, seqlen_k >= 0, batch > 0, nhead > 0.
inline GridDim dqdkdv_grid_size(int batch, int nhead, int seqlen_k, int block_n0)
{
#ifndef NDEBUG
    if(block_n0 <= 0)
    {
        std::fprintf(
            stderr, "rocm_ck::dqdkdv_grid_size: block_n0 must be positive, got %d\n", block_n0);
        std::abort();
    }
    if(seqlen_k < 0)
    {
        std::fprintf(
            stderr, "rocm_ck::dqdkdv_grid_size: seqlen_k must be non-negative, got %d\n", seqlen_k);
        std::abort();
    }
    if(batch <= 0)
    {
        std::fprintf(stderr, "rocm_ck::dqdkdv_grid_size: batch must be positive, got %d\n", batch);
        std::abort();
    }
    if(nhead <= 0)
    {
        std::fprintf(stderr, "rocm_ck::dqdkdv_grid_size: nhead must be positive, got %d\n", nhead);
        std::abort();
    }
#endif
    const auto uk = static_cast<unsigned>(seqlen_k);
    const auto ub = static_cast<unsigned>(block_n0);
    return {(uk + ub - 1u) / ub, static_cast<unsigned>(nhead), static_cast<unsigned>(batch)};
}

// ---------------------------------------------------------------------------
// Debug-only runtime Args validation
// ---------------------------------------------------------------------------

/// Validate that all required tensor and scalar slots for DqDkDv are populated.
/// Compiles to nothing in release builds.
inline void validateArgs([[maybe_unused]] const Args& args, [[maybe_unused]] FmhaBwdDQDKDVSpec k)
{
#ifndef NDEBUG
    namespace S = fmha_bwd_dqdkdv_slots;

    // clang-format off
    static constexpr const char* tensor_names[] = {
        "Q", "K", "V", "LSE", "DO", "D", "DQ_ACC", "DK", "DV",
        "BIAS", "DBIAS", "RANDVAL",
        "SEQSTART_Q", "SEQSTART_K", "SEQLEN_Q", "SEQLEN_K",
        "NSPLITS", "DQ_ACC_BATCH_OFFSET"
    };
    // clang-format on

    int n = S::requiredTensors(k);
    for(int i = 0; i < n; ++i)
    {
        // Skip optional feature slots that are not enabled for this config.
        // Slots 9-11 (BIAS, DBIAS, RANDVAL) are intentionally unpopulated
        // when their respective features are disabled.
        if(i == S::BIAS && k.bias_type == FmhaBiasType::NONE)
            continue;
        if(i == S::DBIAS && !k.has_bias_grad)
            continue;
        // RANDVAL is populated only for dropout-enabled variants.
        if(i == S::RANDVAL && !k.has_dropout)
            continue;
        // Group-mode-only tensor slots: skip in BATCH mode.
        if(k.mode != FmhaMode::GROUP &&
           (i == S::SEQSTART_Q || i == S::SEQSTART_K || i == S::SEQLEN_Q || i == S::SEQLEN_K))
            continue;
        // SEQLEN_Q/SEQLEN_K may be left null in group mode -- CK Tile derives
        // per-batch lengths from SEQSTART_Q/SEQSTART_K when these are absent.
        if(i == S::SEQLEN_Q || i == S::SEQLEN_K)
            continue;

        if(args.tensors[i].ptr == nullptr)
        {
            std::fprintf(stderr,
                         "rocm_ck::validateArgs(DqDkDv): tensor \"%s\" (slot %d)"
                         " has null pointer\n",
                         tensor_names[i],
                         i);
            std::abort();
        }
    }

    // Sanity-check scalar values
    float scale = args.scalars[S::SCALE].f32;
    if(scale == 0.0f)
    {
        std::fprintf(stderr, "rocm_ck::validateArgs(DqDkDv): SCALE (slot %d) is zero\n", S::SCALE);
        std::abort();
    }

    int nhead_ratio = args.scalars[S::NHEAD_RATIO_QK].i32;
    if(nhead_ratio <= 0)
    {
        std::fprintf(stderr,
                     "rocm_ck::validateArgs(DqDkDv): NHEAD_RATIO_QK (slot %d)"
                     " must be positive, got %d\n",
                     S::NHEAD_RATIO_QK,
                     nhead_ratio);
        std::abort();
    }

    // BATCH_SIZE is required only by the deterministic+BATCH persistent kernel
    // path (CK Tile reads kargs.batch for total_jobs); other configs ignore it.
    // Predicate is centralized in usesBatchSizeSlot() so the spec, validator,
    // and device bridge cannot drift apart.
    if(usesBatchSizeSlot(k))
    {
        const int32_t batch_size = args.scalars[S::BATCH_SIZE].i32;
        if(batch_size <= 0)
        {
            std::fprintf(stderr,
                         "rocm_ck::validateArgs(DqDkDv): BATCH_SIZE (slot %d)"
                         " must be positive for deterministic batch mode, got %d\n",
                         S::BATCH_SIZE,
                         batch_size);
            std::abort();
        }
    }
#endif
}

} // namespace rocm_ck
