// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/config.hpp"

namespace ck_tile {

// Architecture tags
struct gfx11_t
{
};
struct gfx12_t
{
};
// Base traits for WMMA operations
template <typename Arch,
          typename AType,
          typename BType,
          typename CType,
          index_t M,
          index_t N,
          index_t K>
struct WmmaTraits;

#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_fp16_traits.hpp"

// Generic WMMA implementation using traits
template <typename Traits>
struct WarpGemmAttributeWmmaImpl
{
    using ADataType = typename Traits::ADataType;
    using BDataType = typename Traits::BDataType;
    using CDataType = typename Traits::CDataType;

    using AVecType = typename Traits::AVecType;
    using BVecType = typename Traits::BVecType;
    using CVecType = typename Traits::CVecType;

    // Forward all static constants and type aliases
    static constexpr index_t kM = Traits::kM;
    static constexpr index_t kN = Traits::kN;
    static constexpr index_t kK = Traits::kK;

    static constexpr index_t kRepeat      = Traits::kRepeat;
    static constexpr index_t kAMLane      = Traits::kAMLane;
    static constexpr index_t kBNLane      = Traits::kBNLane;
    static constexpr index_t kABK0PerLane = Traits::kABK0PerLane;
    static constexpr index_t kABKLane     = Traits::kABKLane;
    static constexpr index_t kABK1PerLane = Traits::kABK1PerLane;

    static constexpr index_t kCMLane     = Traits::kCMLane;
    static constexpr index_t kCNLane     = Traits::kCNLane;
    static constexpr index_t kCM0PerLane = Traits::kCM0PerLane;
    static constexpr index_t kCM1PerLane = Traits::kCM1PerLane;

    using kABPs2RHssMajor = typename Traits::kABPs2RHssMajor;
    using kABPs2RHssMinor = typename Traits::kABPs2RHssMinor;
    using kABYs2RHsMajor  = typename Traits::kABYs2RHsMajor;
    using kABYs2RHsMinor  = typename Traits::kABYs2RHsMinor;

    using kCPs2RHssMajor = typename Traits::kCPs2RHssMajor;
    using kCPs2RHssMinor = typename Traits::kCPs2RHssMinor;
    using kCYs2RHsMajor  = typename Traits::kCYs2RHsMajor;
    using kCYs2RHsMinor  = typename Traits::kCYs2RHsMinor;

    using kCPs2RHssTransMajor = typename Traits::kCPs2RHssTransMajor;
    using kCPs2RHssTransMinor = typename Traits::kCPs2RHssTransMinor;
    using kCYs2RHsTransMajor  = typename Traits::kCYs2RHsTransMajor;
    using kCYs2RHsTransMinor  = typename Traits::kCYs2RHsTransMinor;

    // c_vec += a_vec * b_vec
    template <bool clamp = false, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        c_vec = Traits::template wmma_intrinsic<clamp>(a_vec, b_vec, c_vec);
    }

    // c_vec = a_vec * b_vec
    template <bool clamp = false>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return bit_cast<CVecType>(
            Traits::template wmma_intrinsic<clamp>(a_vec, b_vec, CVecType{0.f}));
    }
};

using WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16_gfx11 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx11_t, fp16_t, fp16_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16_gfx12 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx12_t, fp16_t, fp16_t, float, 16, 16, 16>>;

} // namespace ck_tile
