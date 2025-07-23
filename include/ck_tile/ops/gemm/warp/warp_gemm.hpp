// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"

#include "ck_tile/ops/gemm/warp/warp_gemm_smfmac_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_smfmac.hpp"

namespace ck_tile {

// fp16

using WarpGemmMfmaF16F16F32M32N32K8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfmaF16F16F32M16N16K16 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K16<WGAttrCtlEnum::Default_>>>;

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M32N32K16 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K16<WGAttrCtlEnum::Default_>,
                          AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M32N32K16 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccess>>;
#endif

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M16N16K32 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K32<WGAttrCtlEnum::Default_>,
                          AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M16N16K32 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF16F16F32M16N16K16<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccess>>;
#endif

using WarpGemmMfmaF16F16F32M32N32K8SwizzleA = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK_SwizzleA<
    WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
    1>>;

using WarpGemmMfmaF16F16F32M32N32K16SwizzleA = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK_SwizzleA<
    WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
    2>>;

using WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K16<WGAttrCtlEnum::Default_>>>;

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K16<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccess>>;
#endif

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K32<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K16<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccess>>;
#endif

#if defined(__gfx950__)
using WarpGemmMfmaF16F16F32M16N16K32SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K32<WGAttrCtlEnum::Default_>,
        1>>;

using WarpGemmMfmaBf16Bf16F32M16N16K32SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K32<WGAttrCtlEnum::Default_>,
        1>>;
#endif

#if defined(__gfx950__)
using WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K16<WGAttrCtlEnum::Default_>>>;
#else
using WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2>>;
#endif

using WarpGemmMfmaF16F16F32M4N64K16 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF16F16F32M4N64K4<WGAttrCtlEnum::Default_>,
    4>>;

using WarpGemmMfmaF16F16F32M64N4K16 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF16F16F32M64N4K4<WGAttrCtlEnum::Default_>,
    4>>;

// fp16 2:4 structured sparsity
using WarpGemmSmfmacF16F16F32M32N32K16 = WarpGemmSmfmacImpl<WarpGemmAttributeSmfmac<
    WarpGemmAttributeSmfmacImplF16F16F32M32N32K16<WGAttrCtlEnum::Default_>>>;

using WarpGemmSmfmacF16F16F32M16N16K32 = WarpGemmSmfmacImpl<WarpGemmAttributeSmfmac<
    WarpGemmAttributeSmfmacImplF16F16F32M16N16K32<WGAttrCtlEnum::Default_>>>;

// bf16
using WarpGemmMfmaBf16Bf16F32M32N32K8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfmaBf16Bf16F32M16N16K16 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Default_>>>;

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M32N32K16 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K16<WGAttrCtlEnum::Default_>,
                          AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M32N32K16 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccess>>;
#endif

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M16N16K32 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K32<WGAttrCtlEnum::Default_>,
                          AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M16N16K32 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccess>>;
#endif

using WarpGemmMfmaBf16Bf16F32M32N32K8SwizzleA = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK_SwizzleA<
    WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>,
    1>>;

using WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateK_SwizzleA<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2>>;

using WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Default_>>>;

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K16<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccess>>;
#endif

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K32<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccess>>;
#endif

#if defined(__gfx950__)
using WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K16<WGAttrCtlEnum::Default_>>>;
#else
using WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2>>;
#endif

using WarpGemmMfmaBf16Bf16F32M4N64K16 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImplBf16Bf16F32M4N64K4<WGAttrCtlEnum::Default_>,
    4>>;

using WarpGemmMfmaBf16Bf16F32M64N4K16 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImplBf16Bf16F32M64N4K4<WGAttrCtlEnum::Default_>,
    4>>;

// fp8

using WarpGemmMfma_f32_32x32x16_fp8_fp8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_fp8_bf8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_bf8_fp8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_bf8_bf8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x32_fp8_fp8 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8<WGAttrCtlEnum::Default_>,
    2>>;

using WarpGemmMfma_f32_32x32x32_bf8_bf8 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8<WGAttrCtlEnum::Default_>,
    2>>;

using WarpGemmMfma_f32_16x16x32_fp8_fp8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x32_bf8_bf8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_16x16x32_bf8_bf8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x64_fp8_fp8 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_fp8<WGAttrCtlEnum::Default_>,
    2>>;

using WarpGemmMfma_f32_16x16x64_bf8_bf8 = WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_16x16x32_bf8_bf8<WGAttrCtlEnum::Default_>,
    2>>;

using WarpGemmMfma_f32_16x16x128_fp8_fp8 = WarpGemmImpl<WarpGemmAtrributeMfma<
    WarpGemmAttributeMfmaImpl_f32_16x16x128_fp8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x128_fp8_bf8 = WarpGemmImpl<WarpGemmAtrributeMfma<
    WarpGemmAttributeMfmaImpl_f32_16x16x128_fp8_bf8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x128_bf8_fp8 = WarpGemmImpl<WarpGemmAtrributeMfma<
    WarpGemmAttributeMfmaImpl_f32_16x16x128_bf8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x128_bf8_bf8 = WarpGemmImpl<WarpGemmAtrributeMfma<
    WarpGemmAttributeMfmaImpl_f32_16x16x128_bf8_bf8<WGAttrCtlEnum::Default_>>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_fp8_fp8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x64_fp8_fp8<WGAttrCtlEnum::Default_>,
                          AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_fp8_bf8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x64_fp8_bf8<WGAttrCtlEnum::Default_>,
                          AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_bf8_fp8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x64_bf8_fp8<WGAttrCtlEnum::Default_>,
                          AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_bf8_bf8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x64_bf8_bf8<WGAttrCtlEnum::Default_>,
                          AttrNumAccess>>;

using WarpGemmMfma_f32_32x32x16_fp8_fp8_CTransposed =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_fp8_bf8_CTransposed =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_bf8_fp8_CTransposed =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_bf8_bf8_CTransposed =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8<WGAttrCtlEnum::Default_>>>;

template <index_t swizzle_factor = 2>
using WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<fp8_t, fp8_t, WGAttrCtlEnum::Default_>,
        2,
        swizzle_factor>>;

// int8
using WarpGemmMfma_i32_32x32x16_i8_i8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_i32_32x32x16_i8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_i32_32x32x16_i8_i8_CTransposed =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_i32_32x32x16_i8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_i32_16x16x32_i8_i8 = WarpGemmImpl<
    WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImpl_i32_16x16x32_i8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_i32_16x16x32_i8_i8_CTransposed =
    WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_i32_16x16x32_i8<WGAttrCtlEnum::Default_>>>;

} // namespace ck_tile
