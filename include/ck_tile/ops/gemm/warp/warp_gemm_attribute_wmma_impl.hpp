// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

#if 0
// TODO: refactor warp-gemm
// currently there is a discrepency for vav/vva if we need transpose C/D
// e.g. if we want A:agpr, B:vgpr, we have to use vva in WGAttrEnum
// because we swap the A/B pointer in _impl code (but not known this info here)
enum class WGAttrCtlEnum
{
    Default_ = 0,
    Raw_vvv  = 1, // c-vgpr, a-vgpr, b-vgpr
    Raw_vaa  = 2, // c-vgpr, a-agpr, b-agpr
    Raw_vav  = 3, // c-vgpr, a-agpr, b-vgpr
    Raw_vva  = 4, // c-vgpr, a-vgpr, b-agpr
    Raw_avv  = 5, // c-agpr, a-vgpr, b-vgpr
};
#endif

#if 0

#define DISPATCH_MFMA_(mfma_, dmod_, amod_, bmod_, cmod_)       \
    if constexpr(post_nop_)                                     \
    {                                                           \
        asm volatile(mfma_ " %0, %1, %2, %3 ; yyy\n"            \
                           "s_nop 3"                            \
                     : dmod_(c_vec)                             \
                     : amod_(a_vec), bmod_(b_vec), cmod_(c_vec) \
                     :);                                        \
    }                                                           \
    else                                                        \
    {                                                           \
        asm volatile(mfma_ " %0, %1, %2, %3\n"                  \
                     : dmod_(c_vec)                             \
                     : amod_(a_vec), bmod_(b_vec), cmod_(c_vec) \
                     :);                                        \
    }

#define DISPATCH_MFMA_CTRL_(mfma_, ctrl_)              \
    if constexpr(ctrl_ == WGAttrCtlEnum::Raw_vvv)      \
    {                                                  \
        DISPATCH_MFMA_(mfma_, "+v", "v", "v", "v")     \
    }                                                  \
    else if constexpr(ctrl_ == WGAttrCtlEnum::Raw_vaa) \
    {                                                  \
        DISPATCH_MFMA_(mfma_, "+v", "a", "a", "v")     \
    }                                                  \
    else if constexpr(ctrl_ == WGAttrCtlEnum::Raw_vav) \
    {                                                  \
        DISPATCH_MFMA_(mfma_, "+v", "a", "v", "v")     \
    }                                                  \
    else if constexpr(ctrl_ == WGAttrCtlEnum::Raw_vva) \
    {                                                  \
        DISPATCH_MFMA_(mfma_, "+v", "v", "a", "v")     \
    }                                                  \
    else if constexpr(ctrl_ == WGAttrCtlEnum::Raw_avv) \
    {                                                  \
        DISPATCH_MFMA_(mfma_, "+a", "v", "v", "a")     \
    }

#endif

// FP16
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeWmmaImplF16F16F32M16N16K16
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 8>;
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<float, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
#if 0
        DISPATCH_MFMA_CTRL_("v_mfma_f32_16x16x16f16", Ctrl)
        else
#endif
        {
#if defined(__gfx12__)
            c_vec = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_vec, b_vec, c_vec);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx12__)
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
