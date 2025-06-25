// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/config.hpp"

namespace ck_tile {

// FP16
struct WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 8>;
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<float, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kRepeat      = 1;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 4;
    static constexpr index_t kABKLane     = 2;
    static constexpr index_t kABK1PerLane = 2;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 2;

    static_assert(kCMLane * kCM0PerLane * kCM1PerLane == kM,
                  "Product of kCMLane, kCM0PerLane and kCM1PerLane must equal kM");
    static_assert(kCNLane == kN, "kCNLane must equal kN");
    static_assert(kABK0PerLane * kABKLane * kABK1PerLane == kK,
                  "kK must equal kABK0PerLane * kABKLane * kABK1PerLane");

    using kABPs2RHssMajor = sequence<1, 2>;
    using kABPs2RHssMinor = sequence<0, 1>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<0, 1>;
    using kCYs2RHsMajor  = sequence<2, 2>;
    using kCYs2RHsMinor  = sequence<0, 2>;
    // c_vec += a_vec * b_vec
    template <bool clamp = false, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
#if defined(__gfx13__)
        c_vec = __builtin_amdgcn_wmma_f32_16x16x16_f16_clamp(a_vec, b_vec, c_vec, clamp);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    template <bool clamp = false>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx13__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_wmma_f32_16x16x16_f16_clamp(a_vec, b_vec, fp32x8_t{0.f}, clamp));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

struct WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16_gfx11
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 16>;
    using BVecType = ext_vector_t<fp16_t, 16>;
    using CVecType = ext_vector_t<float, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kRepeat      = 2;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 1;
    static constexpr index_t kABKLane     = 1;
    static constexpr index_t kABK1PerLane = 16;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 8;
    static constexpr index_t kCM1PerLane = 1;

    using kABPs2RHssMajor = sequence<0, 2, 1>;
    using kABPs2RHssMinor = sequence<0, 1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssTransMajor = sequence<2, 1>;
    using kCPs2RHssTransMinor = sequence<1, 0>;
    using kCYs2RHsTransMajor  = sequence<2, 2>;
    using kCYs2RHsTransMinor  = sequence<0, 2>;

    // c_vec += a_vec * b_vec
    template <bool clamp = false, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
#ifdef __gfx11__
        c_vec = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    template <bool clamp = false>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#ifdef __gfx11__
        return bit_cast<CVecType>(
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_vec, b_vec, fp32x8_t{0.f}));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

struct WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16_gfx12
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 8>;
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<float, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kRepeat      = 1;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 2;
    static constexpr index_t kABKLane     = 2;
    static constexpr index_t kABK1PerLane = 4;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 8;

    using kABPs2RHssMajor = sequence<2, 1>;
    using kABPs2RHssMinor = sequence<1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssTransMajor = sequence<2, 1>;
    using kCPs2RHssTransMinor = sequence<1, 0>;
    using kCYs2RHsTransMajor  = sequence<2, 2>;
    using kCYs2RHsTransMinor  = sequence<0, 2>;

    // c_vec += a_vec * b_vec
    template <bool clamp = false, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
#ifdef __gfx12__
        c_vec = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    template <bool clamp = false>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#ifdef __gfx12__
        return bit_cast<CVecType>(
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_vec, b_vec, fp32x8_t{0.f}));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

} // namespace ck_tile
