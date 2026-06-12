// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_traits.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

#include <type_traits>

namespace ck_tile::core::arch::mma {

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Sparse MFMA (SMFMA) on GFX942, GFX950 targets
 *
 * This specialization implements the SMFMA instruction for fp16_t A and B
 * matrices with structured sparsity, fp32_t accumulator, with 16x16x32 fragment sizes.
 *
 * @tparam CtrlFlags Control flags for the Sparse MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsSparseMfmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, std::enable_if_t<is_any_value_of(CompilerTarget::TARGET_ID, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950)>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x32_f16";

    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx) -> CVecType
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x32_f16(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x16_f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x16_f16(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x32_bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x32_bf16(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x16_bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x16_bf16(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX942 and
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_i32_16x16x64_i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_i32_16x16x64_i8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX942 and
 * GFX950 architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_i32_32x32x32_i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_i32_32x32x32_i8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, bf8_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, fp8_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, bf8_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, fp8_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, bf8_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, fp8_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, bf8_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, fp8_t, fp32_t MMA operation on GFX942 and GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsCdna3I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
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
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x64_f16(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
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
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x32_f16(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
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
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x64_bf16(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
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
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x32_bf16(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
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
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_i32_16x16x128_i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_i32_16x16x128_i8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
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
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_i32_32x32x64_i8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_i32_32x32x64_i8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, bf8_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, fp8_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, bf8_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, fp8_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, bf8_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, fp8_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 32u, 32u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, bf8_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 32u, 32u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, fp8_t, fp32_t MMA operation on GFX950
 * architecture.
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx950I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};
} // namespace ck_tile::core::arch::mma
