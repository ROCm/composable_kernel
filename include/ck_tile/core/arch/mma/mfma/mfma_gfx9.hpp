// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "mfma.hpp"

namespace ck_tile::core::arch::mma {
// NOTE: At this point forward, we are specializing for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

// fp16

template <typename CtrlFlags, uint32_t GfxTargetId>
struct amdgcn_mma<float16_t,
                  float16_t,
                  float32_t,
                  16u,
                  16u,
                  16u,
                  CtrlFlags,
                  GfxTargetId,
                  enable_if_gfx9_target_id_t<GfxTargetId>>
{
    using OpType = MfmaOp;

    // Packed register types
    using AVecType = ext_vector_t<float, 2>; // F16x4
    using BVecType = ext_vector_t<float, 2>; // F16x4
    using CVecType = ext_vector_t<float, 4>;

    // Layout constants
    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    CK_TILE_DEVICE static auto
    exec(AVecType const& a_vec, BVecType const& b_vec, CVecType const& c_vec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x16f16(
            a_vec, b_vec, c_vec, (int)CtrlFlags::Cbsz, (int)CtrlFlags::Abid, (int)CtrlFlags::Blgp)};
    }
};

// NOTE: Example here for a specialization on a specific target id
template <typename CtrlFlags, uint32_t GfxTargetId>
struct amdgcn_mma<float16_t,
                  float16_t,
                  float32_t,
                  16u,
                  16u,
                  32u,
                  CtrlFlags,
                  GfxTargetId,
                  enable_if_target_arch_id_t<GfxTargetId, amdgcn_target_arch_id::GFX950>>
{
    using OpType = MfmaOp;

    // Packed register types
    using AVecType = ext_vector_t<float, 4>; // F16x8
    using BVecType = ext_vector_t<float, 4>; // F16x8
    using CVecType = ext_vector_t<float, 4>;

    // Layout constants
    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 8;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    CK_TILE_DEVICE static auto
    exec(AVecType const& a_vec, BVecType const& b_vec, CVecType const& c_vec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_f16(
            a_vec, b_vec, c_vec, (int)CtrlFlags::Cbsz, (int)CtrlFlags::Abid, (int)CtrlFlags::Blgp)};
    }
};

} // namespace ck_tile::core::arch::mma
