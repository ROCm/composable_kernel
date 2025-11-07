// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "warp_gemm_attribute_mfma_impl.hpp"

namespace ck_tile {

// Primary template: generic Smfmac warp-gemm attribute (to be specialized per supported shape)
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          index_t kM,
          index_t kN,
          index_t kK,
          WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeSmfmacImpl; // no definition, only specializations are provided

// fp16 2:4 structured sparsity
template <WGAttrCtlEnum Ctrl_>
struct WarpGemmAttributeSmfmacImpl<fp16_t, fp16_t, float, 32, 32, 16, Ctrl_>
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using IdxDataType                   = int32_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<float, 16>;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    static constexpr index_t CompressionRatio = 2;

    // c_vec += a_vec * b_vec[idx]
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   const int32_t& idx,
                                   bool_constant<post_nop_> = {}) const
    {
#if defined(__gfx94_) or defined(__gfx95_)
        c_vec = __builtin_amdgcn_smfmac_f32_32x32x16_f16(a_vec, b_vec, c_vec, idx, 0, 0);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = idx;
#endif
    }

    // c_vec = a_vec * b_vec[idx]
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec,
                                       const BVecType& b_vec,
                                       const int32_t& idx) const
    {
#if defined(__gfx94_) or defined(__gfx95_)
        return bit_cast<CVecType>(
            __builtin_amdgcn_smfmac_f32_32x32x16_f16(a_vec, b_vec, fp32x4_t{0.f}, idx, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = idx;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_>
struct WarpGemmAttributeSmfmacImpl<fp16_t, fp16_t, float, 16, 16, 32, Ctrl_>
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using IdxDataType                   = int32_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<float, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 32;

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

    static constexpr index_t CompressionRatio = 2;

    // c_vec += a_vec * b_vec[idx]
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   const int32_t& idx,
                                   bool_constant<post_nop_> = {}) const
    {
#if defined(__gfx94_) or defined(__gfx95_)
        c_vec = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a_vec, b_vec, c_vec, idx, 0, 0);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = idx;
#endif
    }

    // c_vec = a_vec * b_vec[idx]
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec,
                                       const BVecType& b_vec,
                                       const int32_t& idx) const
    {
#if defined(__gfx94_) or defined(__gfx95_)
        return bit_cast<CVecType>(
            __builtin_amdgcn_smfmac_f32_16x16x32_f16(a_vec, b_vec, fp32x4_t{0.f}, idx, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = idx;
        return CVecType{0.f};
#endif
    }
};

// Back-compat aliases
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeSmfmacImplF16F16F32M16N16K32 =
    WarpGemmAttributeSmfmacImpl<fp16_t, fp16_t, float, 16, 16, 32, Ctrl_>;

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeSmfmacImplF16F16F32M32N32K16 =
    WarpGemmAttributeSmfmacImpl<fp16_t, fp16_t, float, 32, 32, 16, Ctrl_>;

} // namespace ck_tile
