// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/config.hpp"

namespace ck_tile {

// Base traits for WMMA operations
template <typename Arch,
          typename AType,
          typename BType,
          typename CType,
          index_t M,
          index_t N,
          index_t K,
          typename MXTypeEnable = void>
struct WmmaTraits;

// Tag used to select scale16 WMMA traits specializations.
struct WmmaScale16Tag
{
};

// Generic WMMA implementation using traits
template <typename Traits>
struct WarpGemmAttributeWmmaImpl
{
    using TraitsType = Traits;
    using ADataType  = typename Traits::ADataType;
    using BDataType  = typename Traits::BDataType;
    using CDataType  = typename Traits::CDataType;

    using AVecType = typename Traits::AVecType;
    using BVecType = typename Traits::BVecType;
    using CVecType = typename Traits::CVecType;

    // Forward all static constants and type aliases
    static constexpr index_t kM = Traits::kM;
    static constexpr index_t kN = Traits::kN;
    static constexpr index_t kK = Traits::kK;

    static constexpr index_t kAMBlock = Traits::kAMBlock;
    static constexpr index_t kBNBlock = Traits::kBNBlock;

    static constexpr index_t kCMBlock = Traits::kCMBlock;
    static constexpr index_t kCNBlock = Traits::kCNBlock;

    static constexpr index_t kRepeat     = Traits::kRepeat;
    static constexpr index_t kAMLane     = Traits::kAMLane;
    static constexpr index_t kBNLane     = Traits::kBNLane;
    static constexpr index_t kAK0PerLane = Traits::kAK0PerLane;
    static constexpr index_t kBK0PerLane = Traits::kBK0PerLane;
    static constexpr index_t kAK1PerLane = Traits::kAK1PerLane;
    static constexpr index_t kBK1PerLane = Traits::kBK1PerLane;
    static constexpr index_t kABKLane    = Traits::kABKLane;

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

    using kCTPs2RHssMajor = typename Traits::kCTPs2RHssMajor;
    using kCTPs2RHssMinor = typename Traits::kCTPs2RHssMinor;
    using kCTYs2RHsMajor  = typename Traits::kCTYs2RHsMajor;
    using kCTYs2RHsMinor  = typename Traits::kCTYs2RHsMinor;

    // c_vec += a_vec * b_vec
    template <typename... Params>
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        c_vec = Traits::template wmma_intrinsic<Params...>(a_vec, b_vec, c_vec);
    }

    // c_vec = a_vec * b_vec
    template <typename... Params>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return bit_cast<CVecType>(
            Traits::template wmma_intrinsic<Params...>(a_vec, b_vec, CVecType{0.f}));
    }

    // c_out = a_vec * b_vec + c_vec : fp32 accumulate, narrowed C output (e.g. bf16)
    template <typename... Params>
    CK_TILE_DEVICE auto
    mac_downconvert(const CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        return Traits::template wmma_intrinsic_downconvert<Params...>(a_vec, b_vec, c_vec);
    }

    template <typename... Params, typename AScaleType, typename BScaleType>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const AScaleType& a_scale,
                                   const BVecType& b_vec,
                                   const BScaleType& b_scale) const
    {
        c_vec = Traits::template wmma_intrinsic<Params...>(a_vec, a_scale, b_vec, b_scale, c_vec);
    }

    // c_vec = a_vec * b_vec
    template <typename... Params, typename AScaleType, typename BScaleType>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec,
                                       const AScaleType& a_scale,
                                       const BVecType& b_vec,
                                       const BScaleType& b_scale) const
    {
        return bit_cast<CVecType>(Traits::template wmma_intrinsic<Params...>(
            a_vec, a_scale, b_vec, b_scale, CVecType{0.f}));
    }
};

using DeviceIp = remove_cvref_t<decltype(ck_tile::get_device_arch())>;
using WarpGemmAttributeWmmaImpl_f32_16x16x4_f32 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, float, float, float, 16, 16, 4>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<DeviceIp, fp16_t, fp16_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_bf16_bf16 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<DeviceIp, bf16_t, bf16_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_i32_16x16x16_i8_i8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<DeviceIp, int8_t, int8_t, int32_t, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_f8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx120_t, fp8_t, fp8_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_bf8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx120_t, bf8_t, bf8_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_bf8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx120_t, fp8_t, bf8_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_f8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx120_t, bf8_t, fp8_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x32_f16_f16 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, fp16_t, fp16_t, float, 16, 16, 32>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x32_bf16_bf16 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, bf16_t, bf16_t, float, 16, 16, 32>>;

using WarpGemmAttributeWmmaImpl_bf16_16x16x32_bf16_bf16 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, bf16_t, bf16_t, bf16_t, 16, 16, 32>>;

using WarpGemmAttributeWmmaImpl_i32_16x16x64_i8_i8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, int8_t, int8_t, int32_t, 16, 16, 64>>;

using WarpGemmAttributeWmmaImpl_i32_16x16x64_u8_u8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, uint8_t, uint8_t, int32_t, 16, 16, 64>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x64_f8_f8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, fp8_t, fp8_t, float, 16, 16, 64>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x64_bf8_bf8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, bf8_t, bf8_t, float, 16, 16, 64>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x64_f8_bf8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, fp8_t, bf8_t, float, 16, 16, 64>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x64_bf8_f8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, bf8_t, fp8_t, float, 16, 16, 64>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x128_f8_f8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, fp8_t, fp8_t, float, 16, 16, 128>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x128_bf8_bf8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, bf8_t, bf8_t, float, 16, 16, 128>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x128_f8_bf8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, fp8_t, bf8_t, float, 16, 16, 128>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x128_bf8_f8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, bf8_t, fp8_t, float, 16, 16, 128>>;

using WarpGemmAttributeWmmaImpl_f32_32x16x128_f4 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, pk_fp4_t, pk_fp4_t, float, 32, 16, 128>>;

using WarpGemmAttributeWmmaImpl_f32_32x32x128_f4 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, pk_fp4_t, pk_fp4_t, float, 32, 32, 128>>;

using WarpGemmAttributeWmmaImpl_f32_32x32x128_f4_scale16 = WarpGemmAttributeWmmaImpl<
    WmmaTraits<gfx125_t, pk_fp4_t, pk_fp4_t, float, 32, 32, 128, WmmaScale16Tag>>;

using WarpGemmAttributeWmmaImpl_f16_16x16x64_f8_f8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, fp8_t, fp8_t, fp16_t, 16, 16, 64>>;

using WarpGemmAttributeWmmaImpl_f16_16x16x64_bf8_bf8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, bf8_t, bf8_t, fp16_t, 16, 16, 64>>;

using WarpGemmAttributeWmmaImpl_f16_16x16x64_f8_bf8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, fp8_t, bf8_t, fp16_t, 16, 16, 64>>;

using WarpGemmAttributeWmmaImpl_f16_16x16x64_bf8_f8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, bf8_t, fp8_t, fp16_t, 16, 16, 64>>;

template <typename AType, typename BType>
using WarpGemmAttributeWmmaImpl_f32_16x16x128_f8f6f4 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, AType, BType, float, 16, 16, 128>>;

// WmmaScale16Tag (declared above) is passed as MXTypeEnable to WmmaTraits to select scale16
// specializations. These override kAK1PerLane=16 (-> sequence<4,2,16>) and use int64_t scales
// for V_WMMA_SCALE16_F32_16X16X128_F8F6F4, vs the default layout / int32_t.
template <typename AType, typename BType>
using WarpGemmAttributeWmmaImpl_f32_16x16x128_f8f6f4_scale16 = WarpGemmAttributeWmmaImpl<
    WmmaTraits<gfx125_t, AType, BType, float, 16, 16, 128, WmmaScale16Tag>>;

template <typename AType, typename BType>
using WarpGemmAttributeWmmaImpl_f32_32x32x128_f8f6f4 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx125_t, AType, BType, float, 32, 32, 128>>;

template <typename AType, typename BType>
using WarpGemmAttributeWmmaImpl_f32_32x32x128_f8f6f4_scale16 = WarpGemmAttributeWmmaImpl<
    WmmaTraits<gfx125_t, AType, BType, float, 32, 32, 128, WmmaScale16Tag>>;

template <typename Arch,
          typename AType,
          typename BType,
          typename CType,
          index_t warp_m,
          index_t warp_n,
          index_t warp_k>
struct has_wmma_traits
{
    template <typename T>
    static auto
    test(int) -> decltype(std::declval<
                              typename WmmaTraits<T, AType, BType, CType, warp_m, warp_n, warp_k>::
                                  ADataType>(),
                          std::true_type{});

    template <typename>
    static std::false_type test(...);

    static constexpr bool value = decltype(test<Arch>(0))::value;
};

template <typename Arch,
          typename AType,
          typename BType,
          typename CType,
          index_t warp_m,
          index_t warp_n,
          index_t warp_k>
constexpr bool has_wmma_traits_v =
    has_wmma_traits<Arch, AType, BType, CType, warp_m, warp_n, warp_k>::value;
} // namespace ck_tile
