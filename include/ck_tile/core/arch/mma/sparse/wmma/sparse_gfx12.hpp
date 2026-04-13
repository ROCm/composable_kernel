// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_traits.hpp"

namespace ck_tile::core::arch::mma {

// TODO: c++20 template <CtrlFlagsSparseWmmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx) -> CVecType
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(aVec, bVec, cVec, idx)};
    }
};

} // namespace ck_tile::core::arch::mma
