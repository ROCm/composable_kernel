// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "mfma.hpp"

namespace ck_tile::core::arch::wmma
{
        
    // Enabler for all of gfx12, with binary condition.
    template <uint32_t TargetId, bool Cond = true>
    using enable_gfx12_t
        = enable_if_t<contains_number_v<uint32_t,
                                        TargetId,
                                        amdgcn_target_arch_id::GFX1200,
                                        amdgcn_target_arch_id::GFX1201> && Cond>;

    // NOTE: At this point forward, we are specializing for each target id as needed.
    // This is because some built-ins are only available on certain target ids.
    // We can also do things, such add some padding specializations for when we need to use
    // smaller values of K that aren't directly supported by the built-ins.
    // For flexibility, it is recommended that for each backend wrapper it supports at least
    // one packed register for each input to be able to process smaller K values by padding.

    // fp16

    // gfx12 implementations
    template <uint32_t GfxTargetId>
    struct amdgcn_wmma<float16_t,
                       float16_t,
                       float32_t,
                       16u,
                       16u,
                       4u,
                       GfxTargetId,
                       enable_gfx12_t<GfxTargetId>>
    {
        // Packed register types
        using AVecType = ext_vector_t<float, 1>;
        using BVecType = ext_vector_t<float, 1>;
        using CVecType = ext_vector_t<float, 8>;
        using DVecType = ext_vector_t<float, 8>;

        // Fwding implementation to K=8
        using FwdImpl = amdgcn_wmma<float16_t,
                                    float16_t,
                                    float32_t,
                                    16u,
                                    16u,
                                    8u,
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

    template <uint32_t GfxTargetId>
    struct amdgcn_wmma<float16_t,
                       float16_t,
                       float32_t,
                       16u,
                       16u,
                       8u,
                       GfxTargetId,
                       enable_gfx12_t<GfxTargetId>>
    {
        // Packed register types
        using AVecType = ext_vector_t<float, 2>;
        using BVecType = ext_vector_t<float, 2>;
        using CVecType = ext_vector_t<float, 8>;
        using DVecType = ext_vector_t<float, 8>;

        // Fwding implementation to K=8
        using FwdImpl = amdgcn_wmma<float16_t,
                                    float16_t,
                                    float32_t,
                                    16u,
                                    16u,
                                    16u,
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

    template <uint32_t GfxTargetId>
    struct amdgcn_wmma<float16_t,
                        float16_t,
                        float32_t,
                        16u,
                        16u,
                        16u,
                        GfxTargetId,
                        enable_gfx12_t<GfxTargetId>>
    {

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
            exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
        {
            DRegsT result;
            to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                to_native_vector(regsA), to_native_vector(regsB), to_native_vector(regsC))};
            return result;
        }
    };

} // namespace ck_tile::core::arch::wmma
