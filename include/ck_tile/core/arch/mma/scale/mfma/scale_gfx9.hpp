// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/scale/scale_traits.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for fp8_t A and B
 * matrices with fp32_t accumulator, with 16x16x128 block sizes.
 *
 * @tparam CtrlFlags      Control flags for the Scale MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsScaleMfmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize     |AParams  |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 2, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            static_cast<int>(CtrlFlags::OPSEL_A),
            scale_A,
            static_cast<int>(CtrlFlags::OPSEL_B),
            scale_B)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for bf8_t A and B
 * matrices with fp32_t accumulator, with 16x16x128 block sizes.
 *
 * @tparam CtrlFlags      Control flags for the Scale MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsScaleMfmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize     |AParams  |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 2, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            static_cast<int>(CtrlFlags::OPSEL_A),
            scale_A,
            static_cast<int>(CtrlFlags::OPSEL_B),
            scale_B)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for pk_fp4_t A and B
 * matrices with fp32_t accumulator, with 16x16x128 block sizes.
 *
 * @tparam CtrlFlags      Control flags for the Scale MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsScaleMfmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes          | MNK + WaveSize     |AParams  |BPar |CPar |
struct amdgcn_mma<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp4_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp4_t>,
            static_cast<int>(CtrlFlags::OPSEL_A),
            scale_A,
            static_cast<int>(CtrlFlags::OPSEL_B),
            scale_B)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for fp8_t A and B
 * matrices with fp32_t accumulator, with 32x32x64 block sizes.
 *
 * @tparam CtrlFlags      Control flags for the Scale MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsScaleMfmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize    |AParams  |BPar |CPar  |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 2, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            static_cast<int>(CtrlFlags::OPSEL_A),
            scale_A,
            static_cast<int>(CtrlFlags::OPSEL_B),
            scale_B)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for bf8_t A and B
 * matrices with fp32_t accumulator, with 32x32x64 block sizes.
 *
 * @tparam CtrlFlags      Control flags for the Scale MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsScaleMfmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes    | MNK + WaveSize    |AParams  |BPar |CPar  |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 2, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            static_cast<int>(CtrlFlags::OPSEL_A),
            scale_A,
            static_cast<int>(CtrlFlags::OPSEL_B),
            scale_B)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for pk_fp4_t A and B
 * matrices with fp32_t accumulator, with 32x32x64 block sizes.
 *
 * @tparam CtrlFlags      Control flags for the Scale MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsScaleMfmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes          | MNK + WaveSize    |AParams  |BPar |CPar  |
struct amdgcn_mma<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>> 
: amdgcn_mma_base<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp4_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp4_t>,
            static_cast<int>(CtrlFlags::OPSEL_A),
            scale_A,
            static_cast<int>(CtrlFlags::OPSEL_B),
            scale_B)};
    }
};

} // namespace ck_tile::core::arch::mma
