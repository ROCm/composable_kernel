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
 * @brief Specialization of amdgcn_mma for fp32_t, fp32_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 32u, 64u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 32u, 64u, 1u, 64u, 1, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x1f32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x1f32(bit_cast<fp32_t>(aVec),
                                                     bit_cast<fp32_t>(bVec),
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp32_t, fp32_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 64u, 32u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 64u, 32u, 1u, 64u, 1, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x1f32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x1f32(bit_cast<fp32_t>(aVec),
                                                     bit_cast<fp32_t>(bVec),
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp32_t, fp32_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 16u, 64u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 16u, 64u, 1u, 64u, 1, 1, 4, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x1f32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x1f32(bit_cast<fp32_t>(aVec),
                                                     bit_cast<fp32_t>(bVec),
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp32_t, fp32_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 64u, 16u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 64u, 16u, 1u, 64u, 1, 1, 1, 1, 4, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x1f32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x1f32(bit_cast<fp32_t>(aVec),
                                                     bit_cast<fp32_t>(bVec),
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp32_t, fp32_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize  |AParams  |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 4u, 64u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 4u, 64u, 1u, 64u, 1, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x1f32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_4x4x1f32(bit_cast<fp32_t>(aVec),
                                                   bit_cast<fp32_t>(bVec),
                                                   cVec,
                                                   static_cast<int>(CtrlFlags::Cbsz),
                                                   static_cast<int>(CtrlFlags::Abid),
                                                   static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp32_t, fp32_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize  |AParams |BPar  |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 64u, 4u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 64u, 4u, 1u, 64u, 1, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x1f32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_4x4x1f32(bit_cast<fp32_t>(aVec),
                                                   bit_cast<fp32_t>(bVec),
                                                   cVec,
                                                   static_cast<int>(CtrlFlags::Cbsz),
                                                   static_cast<int>(CtrlFlags::Abid),
                                                   static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp32_t, fp32_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 32u, 32u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 32u, 32u, 2u, 64u, 1, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x2f32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x2f32(bit_cast<fp32_t>(aVec),
                                                     bit_cast<fp32_t>(bVec),
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp32_t, fp32_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 16u, 16u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 16u, 16u, 4u, 64u, 1, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x4f32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x4f32(bit_cast<fp32_t>(aVec),
                                                     bit_cast<fp32_t>(bVec),
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 64u, 4u, 64u, 4, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 32u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 32u, 4u, 64u, 4, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 64u, 4u, 64u, 4, 1, 4, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x4f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x4f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 16u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 16u, 4u, 64u, 4, 1, 1, 1, 4, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x4f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x4f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize  |AParams  |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 4u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 4u, 64u, 4u, 64u, 4, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x4f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
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
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize  |AParams |BPar  |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 4u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 4u, 4u, 64u, 4, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x4f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
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
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 32u, 8u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 32u, 8u, 64u, 4, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x8f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x8f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX9
 * architecture.
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

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x16f16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 64u, 4u, 64u, 4, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_32x32x4i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_32x32x4i8(bit_cast<int32_t>(aVec),
                                                    bit_cast<int32_t>(bVec),
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<int8_t, int8_t, int32_t, 64u, 32u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 64u, 32u, 4u, 64u, 4, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_32x32x4i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_32x32x4i8(bit_cast<int32_t>(aVec),
                                                    bit_cast<int32_t>(bVec),
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 64u, 4u, 64u, 4, 1, 4, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_16x16x4i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_16x16x4i8(bit_cast<int32_t>(aVec),
                                                    bit_cast<int32_t>(bVec),
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 64u, 16u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 64u, 16u, 4u, 64u, 4, 1, 1, 1, 4, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_16x16x4i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_16x16x4i8(bit_cast<int32_t>(aVec),
                                                    bit_cast<int32_t>(bVec),
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize  |AParams  |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 4u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 4u, 64u, 4u, 64u, 4, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_4x4x4i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_4x4x4i8(bit_cast<int32_t>(aVec),
                                                  bit_cast<int32_t>(bVec),
                                                  cVec,
                                                  static_cast<int>(CtrlFlags::Cbsz),
                                                  static_cast<int>(CtrlFlags::Abid),
                                                  static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX9
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize  |AParams |BPar  |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 64u, 4u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 64u, 4u, 4u, 64u, 4, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_4x4x4i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_4x4x4i8(bit_cast<int32_t>(aVec),
                                                  bit_cast<int32_t>(bVec),
                                                  cVec,
                                                  static_cast<int>(CtrlFlags::Cbsz),
                                                  static_cast<int>(CtrlFlags::Abid),
                                                  static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX908 and
 * GFX90a architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 8u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 8u, 64u, 4, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_32x32x8i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_32x32x8i8(bit_cast<int32_t>(aVec),
                                                    bit_cast<int32_t>(bVec),
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX908 and
 * GFX90a architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 16u, 64u, 4, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_16x16x16i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_16x16x16i8(bit_cast<int32_t>(aVec),
                                                     bit_cast<int32_t>(bVec),
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX908 and GFX90a
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 64u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 64u, 2u, 64u, 2, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x2bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x2bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX908 and GFX90a
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 32u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 32u, 2u, 64u, 2, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x2bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x2bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX908 and GFX90a
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 64u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 64u, 2u, 64u, 2, 1, 4, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x2bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x2bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX908 and GFX90a
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 16u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 16u, 2u, 64u, 2, 1, 1, 1, 4, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x2bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x2bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX908 and GFX90a
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize  |AParams  |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 4u, 64u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 4u, 64u, 2u, 64u, 2, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x2bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_4x4x2bf16(aVec,
                                                    bVec,
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX908 and GFX90a
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize  |AParams |BPar  |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 4u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 4u, 2u, 64u, 2, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x2bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_4x4x2bf16(aVec,
                                                    bVec,
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX908 and GFX90a
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 4u, 64u, 2, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX908 and GFX90a
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 8u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 8u, 64u, 2, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x8bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x8bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX90A, GFX942,
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna2I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 64u, 4u, 64u, 4, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4bf16_1k";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4bf16_1k(aVec,
                                                         bVec,
                                                         cVec,
                                                         static_cast<int>(CtrlFlags::Cbsz),
                                                         static_cast<int>(CtrlFlags::Abid),
                                                         static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX90A, GFX942,
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna2I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 32u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 32u, 4u, 64u, 4, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4bf16_1k";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4bf16_1k(aVec,
                                                         bVec,
                                                         cVec,
                                                         static_cast<int>(CtrlFlags::Cbsz),
                                                         static_cast<int>(CtrlFlags::Abid),
                                                         static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX90A, GFX942,
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna2I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 64u, 4u, 64u, 4, 1, 4, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x4bf16_1k";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x4bf16_1k(aVec,
                                                         bVec,
                                                         cVec,
                                                         static_cast<int>(CtrlFlags::Cbsz),
                                                         static_cast<int>(CtrlFlags::Abid),
                                                         static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX90A, GFX942,
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna2I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 16u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 16u, 4u, 64u, 4, 1, 1, 1, 4, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x4bf16_1k";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x4bf16_1k(aVec,
                                                         bVec,
                                                         cVec,
                                                         static_cast<int>(CtrlFlags::Cbsz),
                                                         static_cast<int>(CtrlFlags::Abid),
                                                         static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX90A, GFX942,
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna2I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize  |AParams  |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 4u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 4u, 64u, 4u, 64u, 4, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x4bf16_1k";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_4x4x4bf16_1k(aVec,
                                                       bVec,
                                                       cVec,
                                                       static_cast<int>(CtrlFlags::Cbsz),
                                                       static_cast<int>(CtrlFlags::Abid),
                                                       static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX90A, GFX942,
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna2I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize  |AParams |BPar  |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 4u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 4u, 4u, 64u, 4, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x4bf16_1k";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_4x4x4bf16_1k(aVec,
                                                       bVec,
                                                       cVec,
                                                       static_cast<int>(CtrlFlags::Cbsz),
                                                       static_cast<int>(CtrlFlags::Abid),
                                                       static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX90A, GFX942,
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna2I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar  |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 8u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 8u, 64u, 4, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x8bf16_1k";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(aVec,
                                                         bVec,
                                                         cVec,
                                                         static_cast<int>(CtrlFlags::Cbsz),
                                                         static_cast<int>(CtrlFlags::Abid),
                                                         static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX90A, GFX942,
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna2I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, 64u, 4, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x16bf16_1k";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(aVec,
                                                          bVec,
                                                          cVec,
                                                          static_cast<int>(CtrlFlags::Cbsz),
                                                          static_cast<int>(CtrlFlags::Abid),
                                                          static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp64_t, fp64_t, fp64_t MMA operation on GFX90A, GFX942,
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna2I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize   |AParams |BPar |CPar |
struct amdgcn_mma<fp64_t, fp64_t, fp64_t, 16u, 16u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp64_t, fp64_t, fp64_t, 16u, 16u, 4u, 64u, 1, 1, 1, 1, 1, 4, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f64_16x16x4f64";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        // Note: BLGP flag has another meaning for f64 builtins: BLGP bits [0:2] cause negation of
        // the A, B, and C input matrices respectively (ref. ISA docs for MI300 Instinct)
        return {__builtin_amdgcn_mfma_f64_16x16x4f64(bit_cast<fp64_t>(aVec),
                                                     bit_cast<fp64_t>(bVec),
                                                     cVec,
                                                     0, // CBSZ ignored for f64
                                                     0, // ABID ignored for f64
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX942, GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_16x16x32_i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_16x16x32_i8(bit_cast<int64_t>(aVec),
                                                      bit_cast<int64_t>(bVec),
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX942, GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize    |AParams |BPar |CPar  |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_32x32x16_i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_32x32x16_i8(bit_cast<int64_t>(aVec),
                                                      bit_cast<int64_t>(bVec),
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, bf8_t, fp32_t MMA operation on GFX942, GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(bit_cast<int64_t>(aVec),
                                                           bit_cast<int64_t>(bVec),
                                                           cVec,
                                                           static_cast<int>(CtrlFlags::Cbsz),
                                                           static_cast<int>(CtrlFlags::Abid),
                                                           static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, fp8_t, fp32_t MMA operation on GFX942, GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(bit_cast<int64_t>(aVec),
                                                           bit_cast<int64_t>(bVec),
                                                           cVec,
                                                           static_cast<int>(CtrlFlags::Cbsz),
                                                           static_cast<int>(CtrlFlags::Abid),
                                                           static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, bf8_t, fp32_t MMA operation on GFX942, GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(bit_cast<int64_t>(aVec),
                                                           bit_cast<int64_t>(bVec),
                                                           cVec,
                                                           static_cast<int>(CtrlFlags::Cbsz),
                                                           static_cast<int>(CtrlFlags::Abid),
                                                           static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, fp8_t, fp32_t MMA operation on GFX942, GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bit_cast<int64_t>(aVec),
                                                           bit_cast<int64_t>(bVec),
                                                           cVec,
                                                           static_cast<int>(CtrlFlags::Cbsz),
                                                           static_cast<int>(CtrlFlags::Abid),
                                                           static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, bf8_t, fp32_t MMA operation on GFX942, GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar  |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(bit_cast<int64_t>(aVec),
                                                           bit_cast<int64_t>(bVec),
                                                           cVec,
                                                           static_cast<int>(CtrlFlags::Cbsz),
                                                           static_cast<int>(CtrlFlags::Abid),
                                                           static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, fp8_t, fp32_t MMA operation on GFX942, GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar  |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(bit_cast<int64_t>(aVec),
                                                           bit_cast<int64_t>(bVec),
                                                           cVec,
                                                           static_cast<int>(CtrlFlags::Cbsz),
                                                           static_cast<int>(CtrlFlags::Abid),
                                                           static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, bf8_t, fp32_t MMA operation on GFX942, GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar  |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(bit_cast<int64_t>(aVec),
                                                           bit_cast<int64_t>(bVec),
                                                           cVec,
                                                           static_cast<int>(CtrlFlags::Cbsz),
                                                           static_cast<int>(CtrlFlags::Abid),
                                                           static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, fp8_t, fp32_t MMA operation on GFX942, GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar  |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(bit_cast<int64_t>(aVec),
                                                           bit_cast<int64_t>(bVec),
                                                           cVec,
                                                           static_cast<int>(CtrlFlags::Cbsz),
                                                           static_cast<int>(CtrlFlags::Abid),
                                                           static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_f16(aVec,
                                                       bVec,
                                                       cVec,
                                                       static_cast<int>(CtrlFlags::Cbsz),
                                                       static_cast<int>(CtrlFlags::Abid),
                                                       static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_bf16(aVec,
                                                        bVec,
                                                        cVec,
                                                        static_cast<int>(CtrlFlags::Cbsz),
                                                        static_cast<int>(CtrlFlags::Abid),
                                                        static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar  |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x16_f16(aVec,
                                                       bVec,
                                                       cVec,
                                                       static_cast<int>(CtrlFlags::Cbsz),
                                                       static_cast<int>(CtrlFlags::Abid),
                                                       static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar  |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_f32_32x32x16_bf16(aVec,
                                                        bVec,
                                                        cVec,
                                                        static_cast<int>(CtrlFlags::Cbsz),
                                                        static_cast<int>(CtrlFlags::Abid),
                                                        static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize    |AParams  |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_16x16x64_i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_16x16x64_i8(bit_cast<int32x4_t>(aVec),
                                                      bit_cast<int32x4_t>(bVec),
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes       | MNK + WaveSize    |AParams  |BPar |CPar  |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_32x32x32_i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_mfma_i32_32x32x32_i8(bit_cast<int32x4_t>(aVec),
                                                      bit_cast<int32x4_t>(bVec),
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

} // namespace ck_tile::core::arch::mma
