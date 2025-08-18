// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "wmma.hpp"

namespace ck_tile::core::arch::wmma
{
        
    // Enabler for all of gfx11, with binary condition.
    template <uint32_t TargetId, bool Cond = true>
    using enable_gfx11_t
        = enable_if_t<contains_number_v<uint32_t,
                                        TargetId,
                                        amdgcn_target_arch_id::GFX1100,
                                        amdgcn_target_arch_id::GFX1101,
                                        amdgcn_target_arch_id::GFX1102> && Cond>;


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
    
    // NOTE: At this point forward, we are specializing for each target id as needed.
    // This is because some built-ins are only available on certain target ids.
    // We can also do things, such add some padding specializations for when we need to use
    // smaller values of K that aren't directly supported by the built-ins.
    // For flexibility, it is recommended that for each backend wrapper it supports at least
    // one packed register for each input to be able to process smaller K values by padding.

    // fp16 inputs, f32 accumulation
    template <uint32_t GfxTargetId>
    struct amdgcn_wmma<float16_t,
                float16_t,
                float32_t,
                16u,
                16u,
                4u,
                GfxTargetId,
                enable_gfx9_t<GfxTargetId>>
    {

        // Packed register types (duplicated input / b32 accum)
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
                        enable_gfx11_t<GfxTargetId>>
    {
        // Packed register types (duplicated input / b32 accum)
        using AVecType = ext_vector_t<float, 4>;
        using BVecType = ext_vector_t<float, 4>;
        using CVecType = ext_vector_t<float, 8>;
        using DVecType = ext_vector_t<float, 8>;

        // Fwding implementation to K=16
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
                        enable_gfx11_t<GfxTargetId>>
    {
        // Packed register types (duplicated input / b32 accum)
        using AVecType = ext_vector_t<float, 8>;
        using BVecType = ext_vector_t<float, 8>;
        using CVecType = ext_vector_t<float, 8>;
        using DVecType = ext_vector_t<float, 8>;

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
        exec(AVecType const& a_vec, BVecType const& b_vec, CVecType const& c_vec) -> DVecType
        {
            DRegsT result;
            to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                to_native_vector(regsA), to_native_vector(regsB), to_native_vector(regsC))};
            return result;
        }
    };

} // namespace ck_tile::core::arch::wmma
