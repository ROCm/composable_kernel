// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

namespace ck_tile::core::arch::mma {

struct DefaultSparseWmmaCtrlFlags
{
};

// TODO: c++20 template <CtrlFlagsSparseWmmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
struct amdgcn_mma<fp16_t,
                  fp16_t,
                  fp32_t,
                  16u,
                  16u,
                  32u,
                  CtrlFlags,
                  CompilerTarget,
                  MmaOpFamily::SPARSE,
                  enable_if_target_family_gfx12_t<CompilerTarget>>
{
    using OpType                          = WmmaOp;
    static constexpr MmaOpFamily OpFamily = MmaOpFamily::SPARSE;

    static constexpr index_t ABVecN = 16;

    using AVecType = ext_vector_t<fp16_t, ABVecN>;
    using BVecType = ext_vector_t<fp16_t, ABVecN>;
    using CVecType = ext_vector_t<fp32_t, 8>;

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
        static_assert(CompressedSize == 8);
        // TODO: Compressing A on-the-fly should be OK for now, but we need to validate
        // and evaluate changing this to a transform at a higher level.
        // aVec not being const can cause problems when running multiple intrinsics.
        const int32_t idx = ck_tile::compress_a_impl<fp16_t, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};

        return {__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a_vec_pruned, bVec, cVec, idx)};
    }
};

} // namespace ck_tile::core::arch::mma
