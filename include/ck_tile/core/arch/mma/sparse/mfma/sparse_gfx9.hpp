// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_smfmac_impl.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @struct DefaultSparseMfmaCtrlFlags
 * @brief Default MFMA sparse flags, select (VGPR[srcC][7..0]) if srcC is
 * 16-bit or (VGPR[srcC][15..0]) if srcC is 8-bit.
 */
struct DefaultSparseMfmaCtrlFlags
{
    static constexpr SparseCompressionIndex CompressionIndex = SparseCompressionIndex::FIRST;
};

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>

/**
 * @concept SparseMfmaCtrlFlags
 * @brief Expresses the interface of required members for each CtrlFlags type
 */
template <typename CtrlFlags>
concept SparseMfmaCtrlFlags = requires(CtrlFlags ctrlFlags) {
    // Flag members for sparse MFMA instructions
    { CtrlFlags::CompressionIndex } -> std::convertible_to<SparseCompressionIndex>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Sparse MFMA (SMFMA) on GFX942, GFX950 targets
 *
 * This specialization implements the SMFMA instruction for fp16_t A and B
 * matrices with structured sparsity, fp32_t accumulator, with 16x16x32 block sizes.
 *
 * @tparam CtrlFlags Control flags for the Sparse MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsSparseMfmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
struct amdgcn_mma<
    fp16_t,
    fp16_t,
    fp32_t,
    16u,
    16u,
    32u,
    CtrlFlags,
    CompilerTarget,
    MmaOpFamily::SPARSE,
    std::enable_if_t<is_any_value_of(
        CompilerTarget::TARGET_ID, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950)>>
{
    using OpType                          = MfmaOp;
    static constexpr MmaOpFamily OpFamily = MmaOpFamily::SPARSE;

    static constexpr index_t ABVecN = 8;

    using AVecType = ext_vector_t<fp16_t, ABVecN>;
    using BVecType = ext_vector_t<fp16_t, ABVecN>;
    using CVecType = ext_vector_t<fp32_t, 4>;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    static constexpr index_t kCompressionRatio = 2;

    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t CompressedSize = ABVecN / kCompressionRatio;
        using AVecCompressed                    = ext_vector_t<fp16_t, CompressedSize>;
        static_assert(CompressedSize == 4);
        // TODO: Compressing A on-the-fly should be OK for now, but we need to validate
        // and evaluate changing this to a transform at a higher level.
        // aVec not being const can cause problems when running multiple intrinsics.
        const int32_t idx = ck_tile::compress_a_impl<fp16_t, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {aVec[0], aVec[1], aVec[2], aVec[3]};

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS = getBuiltinParams<CtrlFlags::CompressionIndex>();
        return {__builtin_amdgcn_smfmac_f32_16x16x32_f16(
            a_vec_pruned, bVec, cVec, idx, PARAMS.UseFirstIndex, PARAMS.ByteIndexToOverride)};
    }
};

} // namespace ck_tile::core::arch::mma
