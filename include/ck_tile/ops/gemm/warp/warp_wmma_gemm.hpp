// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_16bit_traits.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_highprec_traits.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_8bit_traits.hpp"

namespace ck_tile {

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x4_f32 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x4_f32,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x16_f16_f16 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x16_bf16_bf16 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf16_bf16,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_i32_16x16x16_i8_i8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_i32_16x16x16_i8_i8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x16_f8_f8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_f8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x16_bf8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_bf8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x32_f16_f16 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x32_f16_f16,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x32_bf16_bf16 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x32_bf16_bf16,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_bf16_16x16x32_bf16_bf16 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_bf16_16x16x32_bf16_bf16,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x16_f8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_bf8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x16_bf8_f8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_f8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_i32_16x16x64_i8_i8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_i32_16x16x64_i8_i8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_i32_16x16x64_u8_u8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_i32_16x16x64_u8_u8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x64_f8_f8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x64_f8_f8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x64_bf8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x64_bf8_bf8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x64_f8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x64_f8_bf8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x64_bf8_f8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x64_bf8_f8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f16_16x16x64_f8_f8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f16_16x16x64_f8_f8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f16_16x16x64_bf8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f16_16x16x64_bf8_bf8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f16_16x16x64_f8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f16_16x16x64_f8_bf8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f16_16x16x64_bf8_f8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f16_16x16x64_bf8_f8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x128_f8_f8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x128_f8_f8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x128_bf8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x128_bf8_bf8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x128_f8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x128_f8_bf8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x128_bf8_f8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x128_bf8_f8,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC = false, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_32x16x128_f4 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_32x16x128_f4,
                                       kTransC,
                                       AttrNumAccess,
                                       AttrNumAccess>>;

template <bool kTransC                      = false,
          WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Default,
          bool IsScale16                    = false>
using WarpGemmWmma_f32_32x32x128_f4 = WarpGemmImpl<
    WarpGemmAttributeWmma<std::conditional_t<IsScale16,
                                             WarpGemmAttributeWmmaImpl_f32_32x32x128_f4_scale16,
                                             WarpGemmAttributeWmmaImpl_f32_32x32x128_f4>,
                          kTransC,
                          AttrNumAccess,
                          AttrNumAccess>>;

template <typename AType,
          typename BType,
          bool kTransC,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Default,
          WGAttrNumAccessEnum AttrNumAccessB = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x128_f8f6f4 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x128_f8f6f4<AType, BType>,
                                       kTransC,
                                       AttrNumAccessA,
                                       AttrNumAccessB>>;

template <typename AType,
          typename BType,
          bool kTransC,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Default,
          WGAttrNumAccessEnum AttrNumAccessB = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_16x16x128_f8f6f4_scale16 = WarpGemmImpl<
    WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x128_f8f6f4_scale16<AType, BType>,
                          kTransC,
                          AttrNumAccessA,
                          AttrNumAccessB>>;

template <typename AType,
          typename BType,
          bool kTransC,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Default,
          WGAttrNumAccessEnum AttrNumAccessB = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_32x32x128_f8f6f4 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_32x32x128_f8f6f4<AType, BType>,
                                       kTransC,
                                       AttrNumAccessA,
                                       AttrNumAccessB>>;

template <typename AType,
          typename BType,
          bool kTransC,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Default,
          WGAttrNumAccessEnum AttrNumAccessB = WGAttrNumAccessEnum::Default>
using WarpGemmWmma_f32_32x32x128_f8f6f4_scale16 = WarpGemmImpl<
    WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_32x32x128_f8f6f4_scale16<AType, BType>,
                          kTransC,
                          AttrNumAccessA,
                          AttrNumAccessB>>;

} // namespace ck_tile
