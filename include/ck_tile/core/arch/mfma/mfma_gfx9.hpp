// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "mfma.hpp"

namespace ck_tile::core::arch::mfma
{
        
    // Enabler for all of gfx9, with binary condition.
    template <uint32_t TargetId, bool Cond = true>
    using enable_gfx9_t
        = enable_if_t<contains_number_v<uint32_t,
                                        TargetId,
                                        amdgcn_target_arch_id::GFX908,
                                        amdgcn_target_arch_id::GFX90A,
                                        amdgcn_target_arch_id::GFX942,
                                        amdgcn_target_arch_id::GFX950> && Cond>;

    // NOTE: At this point forward, we are specializing for each target id as needed.
    // This is because some built-ins are only available on certain target ids.
    // We can also do things, such add some padding specializations for when we need to use
    // smaller values of K that aren't directly supported by the built-ins.
    // For flexibility, it is recommended that for each backend wrapper it supports at least
    // one packed register for each input to be able to process smaller K values by padding.

    // fp16

    // Pads to K=16, all gfx9 targets
    template <uint32_t Cbsz, uint32_t Abid, uint32_t Blgp, uint32_t GfxTargetId>
    struct amdgcn_mfma<float16_t,
                float16_t,
                float32_t,
                16u,
                16u,
                8u,
                Cbsz,
                Abid,
                Blgp,
                GfxTargetId,
                enable_gfx9_t<GfxTargetId>>
    {

        // Packed register types
        using AVecType = ext_vector_t<float, 1>;
        using BVecType = ext_vector_t<float, 1>;
        using CVecType = ext_vector_t<float, 4>;
        using DVecType = ext_vector_t<float, 4>;

        // Fwding implementation to K=16
        using FwdImpl = amdgcn_mfma<float16_t,
                                    float16_t,
                                    float32_t,
                                    16u,
                                    16u,
                                    16u,
                                    Cbsz,
                                    Abid,
                                    Blgp,
                                    GfxTargetId>;

        // Inherits layout constants from FwdImpl
        static constexpr index_t kAMBlock    = FwdImpl::kAMBlock;
        static constexpr index_t kBNBlock    = FwdImpl::kBNBlock;
        static constexpr index_t kAMLane     = FwdImpl::kAMLane;
        static constexpr index_t kBNLane     = FwdImpl::kBNLane;
        static constexpr index_t kABKLane    = FwdImpl::kABKLane;
        static constexpr index_t kABKPerLane = FwdImpl::kABKPerLane;
        static constexpr index_t kCMLane     = FwdImpl::kCMLane;
        static constexpr index_t kCNLane     = FwdImpl::kCNLane;
        static constexpr index_t kCM0PerLane = FwdImpl::kCM0PerLane;
        static constexpr index_t kCM1PerLane = FwdImpl::kCM1PerLane;

        CK_TILE_DEVICE static auto
        exec(AVecType const& a_vec, BVecType const& b_vec, CVecType const& c_vec) -> DVecType
        {
            // Pad with 0s
            return FwdImpl::exec(concat(a_vec, AVecType{0}),
                                    concat(b_vec, BVecType{0}),
                                    forward<CVecType const&>(c_vec));
        }
    };

    template <uint32_t Cbsz, uint32_t Abid, uint32_t Blgp, uint32_t GfxTargetId>
    struct amdgcn_mfma<float16_t,
                float16_t,
                float32_t,
                16u,
                16u,
                16u,
                Cbsz,
                Abid,
                Blgp,
                GfxTargetId,
                enable_gfx9_t<GfxTargetId>>
    {
        // Packed register types
        using AVecType = ext_vector_t<float, 2>; // F16x4
        using BVecType = ext_vector_t<float, 2>; // F16x4
        using CVecType = ext_vector_t<float, 4>;
        using DVecType = ext_vector_t<float, 4>;

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
        exec(AVecType const& a_vec, BVecType const& b_vec, CVecType const& c_vec) -> DVecType
        {
            DVecType result;
            to_native_vector(result)
                = {__builtin_amdgcn_mfma_f32_16x16x16f16(to_native_vector(a_vec),
                                    to_native_vector(b_vec),
                                    to_native_vector(c_vec),
                                    (int)Cbsz,
                                    (int)Abid,
                                    (int)Blgp)};
            return result;
        }
    };


    // NOTE: Example here for a specialization on a specific target id
    template <uint32_t Cbsz, uint32_t Abid, uint32_t Blgp, uint32_t GfxTargetId>
    struct amdgcn_mfma<float16_t,
                float16_t,
                float32_t,
                16u,
                16u,
                32u,
                Cbsz,
                Abid,
                Blgp,
                GfxTargetId,
                enable_target_id_t<GfxTargetId, amdgcn_target_arch_id::GFX950>>
    {
        // Packed register types
        using AVecType = ext_vector_t<float, 4>; // F16x8
        using BVecType = ext_vector_t<float, 4>; // F16x8
        using CVecType = ext_vector_t<float, 4>;
        using DVecType = ext_vector_t<float, 4>;

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
        exec(AVecType const& a_vec, BVecType const& b_vec, CVecType const& c_vec) -> DVecType
        {
            DVecType result;
            to_native_vector(result)
                = {__builtin_amdgcn_mfma_f32_16x16x32_f16(to_native_vector(a_vec),
                                    to_native_vector(b_vec),
                                    to_native_vector(c_vec),
                                    (int)Cbsz,
                                    (int)Abid,
                                    (int)Blgp)};
            return result;
        }
    };

} // namespace ck_tile::core::arch::mfma
