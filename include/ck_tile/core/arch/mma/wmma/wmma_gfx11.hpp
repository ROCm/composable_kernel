// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "wmma_traits.hpp"

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

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

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX11
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx11I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams  |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx11_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, 32u, 16, 1, 2, 1, 2, 8, 8, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aVec, bVec, cVec)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX11
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx11I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams  |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx11_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, 32u, 16, 1, 2, 1, 2, 8, 8, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(aVec, bVec, cVec)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX11
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx11I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize    |AParams  |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx11_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 16u, 32u, 16, 1, 2, 1, 2, 8, 8, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, // A signedness
                                                           bit_cast<int32x4_t>(aVec),
                                                           true, // B signedness
                                                           bit_cast<int32x4_t>(bVec),
                                                           cVec,
                                                           CtrlFlags::Clamp)};
    }
};

} // namespace ck_tile::core::arch::mma
