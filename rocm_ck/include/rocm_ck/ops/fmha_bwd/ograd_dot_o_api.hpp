// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-only helpers for the FMHA BWD OGradDotO kernel family.
//
// HOST ONLY: this header must NOT be included from device code (.hip files).
// Device code should include <rocm_ck/ops/fmha_bwd/ograd_dot_o_dev.hpp>.
//
// Compilation boundary:
//   _spec.hpp -- consteval factory + slot constants (both passes)
//   _api.hpp (this) -- host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp -- CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#error "ograd_dot_o_api.hpp is host-only." \
       " Device code should include <rocm_ck/ops/fmha_bwd/ograd_dot_o_dev.hpp>."
#endif

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>

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

/// Compute the launch grid for OGradDotO.
/// Matches FmhaBwdOGradDotOSpec::GridSize():
///   GridDim(ceil(seqlen_q / kM0), nhead, batch).
/// Precondition: block_size > 0, seqlen_q >= 0, batch > 0, nhead > 0.
inline GridDim ograd_dot_o_grid_size(int batch, int nhead, int seqlen_q, int block_size)
{
#ifndef NDEBUG
    if(block_size <= 0)
    {
        std::fprintf(stderr,
                     "rocm_ck::ograd_dot_o_grid_size: block_size must be positive, got %d\n",
                     block_size);
        std::abort();
    }
    if(seqlen_q < 0)
    {
        std::fprintf(stderr,
                     "rocm_ck::ograd_dot_o_grid_size: seqlen_q must be non-negative, got %d\n",
                     seqlen_q);
        std::abort();
    }
    if(batch <= 0)
    {
        std::fprintf(
            stderr, "rocm_ck::ograd_dot_o_grid_size: batch must be positive, got %d\n", batch);
        std::abort();
    }
    if(nhead <= 0)
    {
        std::fprintf(
            stderr, "rocm_ck::ograd_dot_o_grid_size: nhead must be positive, got %d\n", nhead);
        std::abort();
    }
#endif
    const auto uq = static_cast<unsigned>(seqlen_q);
    const auto ub = static_cast<unsigned>(block_size);
    return {(uq + ub - 1u) / ub, static_cast<unsigned>(nhead), static_cast<unsigned>(batch)};
}

// ---------------------------------------------------------------------------
// Debug-only runtime Args validation
// ---------------------------------------------------------------------------

/// Validate that all required tensor slots for OGradDotO are populated.
/// Compiles to nothing in release builds.
inline void validateArgs([[maybe_unused]] const Args& args, [[maybe_unused]] FmhaBwdOGradDotOSpec k)
{
#ifndef NDEBUG
    namespace S = fmha_bwd_ograd_dot_o_slots;

    static constexpr const char* tensor_names[] = {"O", "DO", "D", "SEQSTART_Q", "SEQLEN_Q"};

    int n = S::requiredTensors(k);
    for(int i = 0; i < n; ++i)
    {
        if(args.tensors[i].ptr == nullptr)
        {
            std::fprintf(stderr,
                         "rocm_ck::validateArgs(OGradDotO): tensor \"%s\" (slot %d)"
                         " has null pointer\n",
                         tensor_names[i],
                         i);
            std::abort();
        }
    }

    // Sanity-check scalar values
    float p_undrop = args.scalars[S::P_UNDROP].f32;
    if(p_undrop == 0.0f)
    {
        std::fprintf(
            stderr, "rocm_ck::validateArgs(OGradDotO): P_UNDROP (slot %d) is zero\n", S::P_UNDROP);
        std::abort();
    }
#endif
}

} // namespace rocm_ck
