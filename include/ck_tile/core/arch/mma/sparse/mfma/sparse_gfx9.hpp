// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_traits.hpp"

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
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx) -> CVecType
    {
        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x32_f16(
            aVec, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

} // namespace ck_tile::core::arch::mma
