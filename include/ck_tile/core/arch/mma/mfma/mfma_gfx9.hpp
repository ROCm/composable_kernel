// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "mfma_traits.hpp"

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

namespace ck_tile::core::arch::mma {

// NOTE: At this point forward, we are specializing amdgcn_mma for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for MFMA on GFX9 targets
 *
 * This specialization implements the MFMA instruction for fp16_t A and B
 * matrices, and fp32_t accumulator matrix, with 16x16x16 fragment sizes.
 *
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, 64u, 4, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x16f16";

    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x16f16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 32u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 32u, 4u, 64u, 4, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4f16";

    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 64u, 4u, 64u, 4, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4f16";

    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize  |AParams |BPar  |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 4u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 4u, 4u, 64u, 4, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x4f16";

    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_4x4x4f16(aVec,
                                                   bVec,
                                                   cVec,
                                                   static_cast<int>(CtrlFlags::Cbsz),
                                                   static_cast<int>(CtrlFlags::Abid),
                                                   static_cast<int>(CtrlFlags::Blgp))};
    }
};

template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize  |AParams  |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 4u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 4u, 64u, 4u, 64u, 4, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x4f16";

    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_4x4x4f16(aVec,
                                                   bVec,
                                                   cVec,
                                                   static_cast<int>(CtrlFlags::Cbsz),
                                                   static_cast<int>(CtrlFlags::Abid),
                                                   static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for MFMA on GFX950 targets
 *
 * This specialization implements the MFMA instruction for fp16_t A and B
 * matrices, and fp32_t accumulator matrix, with 16x16x32 fragment sizes.
 *
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_f16";

    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_f16(aVec,
                                                       bVec,
                                                       cVec,
                                                       static_cast<int>(CtrlFlags::Cbsz),
                                                       static_cast<int>(CtrlFlags::Abid),
                                                       static_cast<int>(CtrlFlags::Blgp))};
    }
};

} // namespace ck_tile::core::arch::mma
