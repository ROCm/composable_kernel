// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "warp_gemm_attribute_wmma_impl_base_traits.hpp"
#include "warp_gemm_params.hpp"

namespace ck_tile {
// f32 specialization - GFX125
template <>
struct WmmaTraits<gfx125_t, float, float, float, 16, 16, 4>
    : WmmaTraitsBase<gfx12_t, float, float, float, 4>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x4_f32(
            0, a_vec, 0, b_vec, 0, c_vec, P::reuse_a, P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// TODO: Add f64 WMMA Instruction
// f64 specialization - GFX125

} // namespace ck_tile
