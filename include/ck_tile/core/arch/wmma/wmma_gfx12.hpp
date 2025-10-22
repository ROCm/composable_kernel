// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "wmma.hpp"

namespace ck_tile::core::arch::mma {

// NOTE: At this point forward, we are specializing for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things, such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

// fp16 inputs, f32 accumulation
template <typename CtrlFlags, uint32_t GfxTargetId>
struct amdgcn_wmma<float16_t,
                   float16_t,
                   float32_t,
                   16u,
                   16u,
                   16u,
                   CtrlFlags,
                   GfxTargetId,
                   enable_if_gfx12_target_id_t<GfxTargetId>>
{
    using OpType = WmmaOp;

    // Packed register types
    using AVecType = ext_vector_t<float, 4>;
    using BVecType = ext_vector_t<float, 4>;
    using CVecType = ext_vector_t<float, 8>;
    using DVecType = ext_vector_t<float, 8>;

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
    exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> CRegsT
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(regsA, regsB, regsC)};
    }
};

} // namespace ck_tile::core::arch::mma
