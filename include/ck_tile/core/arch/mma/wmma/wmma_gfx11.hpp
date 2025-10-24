// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "wmma.hpp"

namespace ck_tile::core::arch::mma {
// TODO: Specifically for gfx11 wmma, we need to deal with quirks such as:
//       - Duplicating A and B inputs
//       - Handling C / D is always in b32, even for f16 accumulation.
// NOTE: Two suggestions:
// 1) We could do it here in the wrappers by accepting packed inputs, then swizzling them to
//    duplicate the inputs as needed before calling the actual built-in. This may introduce
//    some instruction overhead and violate single responsibility clauses, but keeps the logic
//    contained within the backend wrapper.
// 2) We could do it at a higher level, e.g. in the Mma interface (workflow) by introducing
//    pre-mma, mma and post-mma steps. The pre-mma step could handle input duplication transform
//    post-mma could implement D-shuffle transform. This may be cleaner and more flexible than
//    trying to handle everything in the backend wrappers.
//
// This current example assumes duplication has already been done, and that C data shuffles have
// already been completed. (e.g. option 2 above). These expect duplicated inputs and pre-shuffled
// data in C.

// NOTE: At this point forward, we are specializing amdgcn_mma for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things, such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

/*! @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX11
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam GfxTargetId Graphics target identifier
 */
template <typename CtrlFlags, uint32_t GfxTargetId>
struct amdgcn_mma<fp16_t,
                  fp16_t,
                  fp32_t,
                  16u,
                  16u,
                  16u,
                  CtrlFlags,
                  GfxTargetId,
                  enable_if_gfx11_target_id_t<GfxTargetId>>
{
    // Wmma operation type
    using OpType = WmmaOp;

    // Register types (duplicated input / b32 accum)
    using AVecType = ext_vector_t<fp16_t, 16>;
    using BVecType = ext_vector_t<fp16_t, 16>;
    using CVecType = ext_vector_t<fp32_t, 8>;

    // Layout constants
    static constexpr index_t kAMBlock    = 1;
    static constexpr index_t kBNBlock    = 1;
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 8;
    static constexpr index_t kABKPerLane = 8;
    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 2;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 1;

    CK_TILE_DEVICE static auto
    exec(AVecType const& a_vec, BVecType const& b_vec, CVecType const& c_vec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(regsA, regsB, regsC)};
    }
};

} // namespace ck_tile::core::arch::mma
