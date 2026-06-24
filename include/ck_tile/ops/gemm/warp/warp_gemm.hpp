// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"

#include "ck_tile/ops/gemm/warp/warp_gemm_smfmac_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_smfmac.hpp"

#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher_unification.hpp"

namespace ck_tile {

// For the pipelines that do not use the dispatcher but instead directly used named WarpGemms,
// redefine them in terms of the new unification dispatcher. Only did so for named WarpGemms that
// are already (or seem likely to be) used directly.
#if USE_NEW_UNIFIED_FRAMEWORK
// fp16 named WarpGemms (no WMMA or StructuredSparsity)
// clang-format off
template<WGAttrNumAccessEnum NumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M32N32K16                                  = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 32, 32, 16, false, false, false, NumAccess>::Type;
template<WGAttrNumAccessEnum NumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution           = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 32, 32, 16, true, false, false, NumAccess>::Type;
template<WGAttrNumAccessEnum NumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M16N16K32                                  = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 16, 16, 32, false, false, false, NumAccess>::Type;
template<WGAttrNumAccessEnum NumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution           = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 16, 16, 32, true, false, false, NumAccess>::Type;
using WarpGemmMfmaF16F16F32M32N32K8                                   = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 32, 32,  8, false>::Type;
using WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution            = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 32, 32,  8,  true>::Type;
using WarpGemmMfmaF16F16F32M4N64K16                                   = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float,  4, 64, 16, false>::Type;
using WarpGemmMfmaF16F16F32M64N4K16                                   = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 64,  4, 16, false>::Type;
using WarpGemmMfmaF16F16F32M16N16K16                                  = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 16, 16, 16, false>::Type;
using WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution           = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 16, 16, 16,  true>::Type;
// using WarpGemmMfmaF16F16F32M32N32K8SwizzleA                           = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 32, 32,  8, false, true>::Type;
// using WarpGemmMfmaF16F16F32M32N32K16SwizzleA                          = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 32, 32, 16, false, true>::Type;
// using WarpGemmMfmaF16F16F32M32N32K8SwizzleBTransposedCDistribution    = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 32, 32,  8,  true, true>::Type;
// using WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution   = typename impl::warp_gemm_dispatcher::UnificationDispatcher<half_t, half_t, float, 32, 32, 16,  true, true>::Type;

// bf16 named WarpGemms (no WMMA or StructuredSparsity)
using WarpGemmMfmaBf16Bf16F32M32N32K8                                 = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 32, 32,  8, false>::Type;
using WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution          = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 32, 32,  8,  true>::Type;
template<WGAttrNumAccessEnum NumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M32N32K16                                = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 32, 32, 16, false, false, false, NumAccess>::Type;
template<WGAttrNumAccessEnum NumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution         = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 32, 32, 16,  true, false, false, NumAccess>::Type;
template<WGAttrNumAccessEnum NumAccessA = WGAttrNumAccessEnum::Single, WGAttrNumAccessEnum NumAccessB = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M16N16K32                                = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 16, 16, 32, false, false, false, NumAccessA, NumAccessB>::Type;
template<WGAttrNumAccessEnum NumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution         = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 16, 16, 32,  true, false, false, NumAccess>::Type;
template<WGAttrNumAccessEnum NumAccessA = WGAttrNumAccessEnum::Single, WGAttrNumAccessEnum NumAccessB = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M16N16K64                                = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 16, 16, 64, false, false, false, NumAccessA, NumAccessB>::Type;
using WarpGemmMfmaBf16Bf16F32M4N64K16                                 = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float,  4, 64, 16, false>::Type;
using WarpGemmMfmaBf16Bf16F32M64N4K16                                 = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 64,  4, 16, false>::Type;
using WarpGemmMfmaBf16Bf16F32M16N16K16                                = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 16, 16, 16, false>::Type;
using WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution         = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 16, 16, 16,  true>::Type;
// using WarpGemmMfmaBf16Bf16F32M32N32K8SwizzleA                         = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 32, 32,  8, false, true>::Type;
// using WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA                        = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 32, 32, 16, false, true>::Type;
// using WarpGemmMfmaBf16Bf16F32M32N32K8SwizzleBTransposedCDistribution  = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 32, 32,  8,  true, true>::Type;
// using WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution = typename impl::warp_gemm_dispatcher::UnificationDispatcher<bf16_t, bf16_t, float, 32, 32, 16,  true, true>::Type;
//clang-format on

#else // #if USE_NEW_UNIFIED_FRAMEWORK

// fp32

using WarpGemmMfmaF32F32F32M16N16K4 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplF32F32F32M16N16K4<WGAttrCtlEnum::Default_>>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF32F32F32M16N16K16 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF32F32F32M16N16K4<WGAttrCtlEnum::Default_>,
    4,
    AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF32F32F32M16N16K8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF32F32F32M16N16K4<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF32F32F32M32N32K8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF32F32F32M32N32K2<WGAttrCtlEnum::Default_>,
    4,
    AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF32F32F32M32N32K4 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF32F32F32M32N32K2<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF32F32F32M16N16K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplF32F32F32M16N16K4<WGAttrCtlEnum::Default_>,
        4,
        AttrNumAccess>>;

// tf32
// On gfx950: uses 3x bf16 MFMA emulation (no native xf32 support)

#if defined(CK_GFX950_SUPPORT)
// gfx950: tf32 emulated using 3x bf16 MFMA
using WarpGemmMfmaTf32Tf32F32M32N32K16Native = WarpGemmImpl<WarpGemmAttributeMfma<
    WarpGemmAttributeMfmaImplF32F32F32M32N32K16Tf32Gfx950<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfmaTf32Tf32F32M16N16K32Native = WarpGemmImpl<WarpGemmAttributeMfma<
    WarpGemmAttributeMfmaImplF32F32F32M16N16K32Tf32Gfx950<WGAttrCtlEnum::Default_>>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaTf32Tf32F32M32N32K16 = WarpGemmImpl<WarpGemmAttributeMfma<
    WarpGemmAttributeMfmaImplF32F32F32M32N32K16Tf32Gfx950<WGAttrCtlEnum::Default_>,
    AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaTf32Tf32F32M16N16K32 = WarpGemmImpl<WarpGemmAttributeMfma<
    WarpGemmAttributeMfmaImplF32F32F32M16N16K32Tf32Gfx950<WGAttrCtlEnum::Default_>,
    AttrNumAccess>>;
#endif

// fp16

using WarpGemmMfmaF16F16F32M32N32K8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfmaF16F16F32M16N16K16 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K16<WGAttrCtlEnum::Default_>>>;

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfmaF16F16F32M32N32K16 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K16<WGAttrCtlEnum::Default_>,
                          AttrNumAccessA,
                          AttrNumAccessB>>;
#else
template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfmaF16F16F32M32N32K16 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccessA,
    AttrNumAccessB>>;
#endif

#if defined(__gfx950__)

template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfmaF16F16F32M16N16K32 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K32<WGAttrCtlEnum::Default_>,
                          AttrNumAccessA,
                          AttrNumAccessB>>;
#else
template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfmaF16F16F32M16N16K32 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF16F16F32M16N16K16<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccessA,
    AttrNumAccessB>>;
#endif

using WarpGemmMfmaF16F16F32M32N32K8SwizzleA = WarpGemmImpl<WarpGemmAttributeMfmaIterateK_SwizzleA<
    WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
    1>>;

using WarpGemmMfmaF16F16F32M32N32K16SwizzleA = WarpGemmImpl<WarpGemmAttributeMfmaIterateK_SwizzleA<
    WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
    2>>;

using WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K16<WGAttrCtlEnum::Default_>>>;

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K16<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccess>>;
#endif

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K32<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K16<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccess>>;
#endif

#if defined(__gfx950__)
using WarpGemmMfmaF16F16F32M16N16K32SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K32<WGAttrCtlEnum::Default_>,
        1>>;

using WarpGemmMfmaBf16Bf16F32M16N16K32SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K32<WGAttrCtlEnum::Default_>,
        1>>;
#endif

using WarpGemmMfmaF16F16F32M32N32K8SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>>>;

#if defined(__gfx950__)
using WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K16<WGAttrCtlEnum::Default_>>>;
#else
using WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2>>;
#endif

using WarpGemmMfmaF16F16F32M4N64K16 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF16F16F32M4N64K4<WGAttrCtlEnum::Default_>,
    4>>;

using WarpGemmMfmaF16F16F32M64N4K16 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplF16F16F32M64N4K4<WGAttrCtlEnum::Default_>,
    4>>;

// fp16 2:4 structured sparsity
using WarpGemmSmfmacF16F16F32M32N32K16 = WarpGemmSmfmacImpl<WarpGemmAttributeSmfmac<
    WarpGemmAttributeSmfmacImplF16F16F32M32N32K16<WGAttrCtlEnum::Default_>>>;

using WarpGemmSmfmacF16F16F32M16N16K32 = WarpGemmSmfmacImpl<WarpGemmAttributeSmfmac<
    WarpGemmAttributeSmfmacImplF16F16F32M16N16K32<WGAttrCtlEnum::Default_>>>;

// bf16
using WarpGemmMfmaBf16Bf16F32M32N32K8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfmaBf16Bf16F32M16N16K16 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Default_>>>;

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfmaBf16Bf16F32M32N32K16 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K16<WGAttrCtlEnum::Default_>,
                          AttrNumAccessA,
                          AttrNumAccessB>>;
#else
template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfmaBf16Bf16F32M32N32K16 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccessA,
    AttrNumAccessB>>;
#endif

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfmaBf16Bf16F32M16N16K32 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K32<WGAttrCtlEnum::Default_>,
                          AttrNumAccessA,
                          AttrNumAccessB>>;

template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfmaBf16Bf16F32M16N16K64 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K32<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccessA,
    AttrNumAccessB>>;
#else
template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfmaBf16Bf16F32M16N16K32 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccessA>>;

template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfmaBf16Bf16F32M16N16K64 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Default_>,
    4,
    AttrNumAccessA,
    AttrNumAccessB>>;
#endif

using WarpGemmMfmaBf16Bf16F32M32N32K8SwizzleA = WarpGemmImpl<WarpGemmAttributeMfmaIterateK_SwizzleA<
    WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>,
    1>>;

using WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateK_SwizzleA<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2>>;

using WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Default_>>>;

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K16<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccess>>;
#endif

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K32<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccess>>;
#endif

using WarpGemmMfmaBf16Bf16F32M32N32K8SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>>>;

#if defined(__gfx950__)
using WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K16<WGAttrCtlEnum::Default_>>>;
#else
using WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2>>;
#endif

using WarpGemmMfmaBf16Bf16F32M4N64K16 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplBf16Bf16F32M4N64K4<WGAttrCtlEnum::Default_>,
    4>>;

using WarpGemmMfmaBf16Bf16F32M64N4K16 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImplBf16Bf16F32M64N4K4<WGAttrCtlEnum::Default_>,
    4>>;

// fp8
#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x32_fp8_fp8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x32_fp8_fp8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8<WGAttrCtlEnum::Default_>,
    2>>;
#endif

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_16x16x64_fp8_fp8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_fp8<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_16x16x64_fp8_fp8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_fp8<WGAttrCtlEnum::Default_>,
    2>>;
#endif

#if defined(__gfx950__)
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x32_bf8_bf8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8<WGAttrCtlEnum::Default_>,
    2,
    AttrNumAccess>>;
#else
template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x32_bf8_bf8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8<WGAttrCtlEnum::Default_>,
    2>>;
#endif

using WarpGemmMfma_f32_32x32x16_fp8_fp8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_fp8_bf8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x32_fp8_bf8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_bf8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_bf8_fp8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_bf8_bf8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8<WGAttrCtlEnum::Default_>>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8<WGAttrCtlEnum::Default_>,
        2,
        AttrNumAccess>>;

using WarpGemmMfma_f32_32x32x32_fp8_bf8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8<WGAttrCtlEnum::Default_>,
    2>>;

using WarpGemmMfma_f32_16x16x32_fp8_fp8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x32_fp8_fp8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x32_bf8_bf8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_16x16x32_bf8_bf8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x32_bf8_fp8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_16x16x32_bf8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x32_bf8_bf8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_16x16x32_bf8_bf8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_16x16x64_bf8_bf8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
    WarpGemmAttributeMfmaImpl_f32_16x16x32_bf8_bf8<WGAttrCtlEnum::Default_>,
    2>>;

using WarpGemmMfma_f32_16x16x64_fp8_fp8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_fp8<WGAttrCtlEnum::Default_>,
        2>>;

using WarpGemmMfma_f32_16x16x64_bf8_bf8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_16x16x32_bf8_bf8<WGAttrCtlEnum::Default_>,
        2>>;

template <typename A,
          typename B,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfma_f32_16x16x128_f8f6f4 =
    WarpGemmImpl<WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_16x16x128_f8f6f4<A, B>,
                                       AttrNumAccessA,
                                       AttrNumAccessB>>;

template <typename A, typename B, WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_16x16x128_f8f6f4_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_16x16x128_f8f6f4<A, B>,
        AttrNumAccess>>;

template <typename A,
          typename B,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfma_f32_32x32x64_f8f6f4 =
    WarpGemmImpl<WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_32x32x64_f8f6f4<A, B>,
                                       AttrNumAccessA,
                                       AttrNumAccessB>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_fp8_fp8 =
    WarpGemmMfma_f32_32x32x64_f8f6f4<fp8_t, fp8_t, AttrNumAccess>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_fp8_bf8 =
    WarpGemmMfma_f32_32x32x64_f8f6f4<fp8_t, bf8_t, AttrNumAccess>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_bf8_fp8 =
    WarpGemmMfma_f32_32x32x64_f8f6f4<bf8_t, fp8_t, AttrNumAccess>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_bf8_bf8 =
    WarpGemmMfma_f32_32x32x64_f8f6f4<bf8_t, bf8_t, AttrNumAccess>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_fp8_fp8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x64_fp8_fp8<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_fp8_bf8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x64_fp8_bf8<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_bf8_fp8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x64_bf8_fp8<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_bf8_bf8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x64_bf8_bf8<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_fp4_fp4_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x64_f8f6f4<pk_fp4_t, pk_fp4_t>,
        AttrNumAccess>>;

template <WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfma_f32_32x32x64_fp4_fp6_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x64_f8f6f4<pk_fp4_t, pk_fp6x16_t>,
        AttrNumAccess>>;

using WarpGemmMfma_f32_32x32x16_fp8_fp8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_fp8_bf8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_bf8_fp8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_fp8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_f32_32x32x16_bf8_bf8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8<WGAttrCtlEnum::Default_>>>;

template <index_t swizzle_factor = 2>
using WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8<WGAttrCtlEnum::Default_>,
        2,
        swizzle_factor>>;

// int8
using WarpGemmMfma_i32_32x32x16_i8_i8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_i32_32x32x16_i8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_i32_32x32x16_i8_i8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_i32_32x32x16_i8<WGAttrCtlEnum::Default_>>>;

using WarpGemmMfma_i32_16x16x32_i8_i8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_i32_16x16x32_i8<WGAttrCtlEnum::Default_>>>;

template <WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA>
using WarpGemmMfma_i32_16x16x64_i8_i8 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_i32_16x16x64_i8<WGAttrCtlEnum::Default_>,
                          AttrNumAccessA,
                          AttrNumAccessB>>;

template <WGAttrNumAccessEnum AttrNumAccess>
using WarpGemmMfma_i32_16x16x64_i8_i8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_i32_16x16x64_i8<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;

using WarpGemmMfma_i32_16x16x32_i8_i8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_i32_16x16x32_i8<WGAttrCtlEnum::Default_>>>;

template <WGAttrNumAccessEnum AttrNumAccess>
using WarpGemmMfma_i32_16x16x64_i8_i8_CTransposed =
    WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
        WarpGemmAttributeMfmaImpl_i32_16x16x64_i8<WGAttrCtlEnum::Default_>,
        AttrNumAccess>>;

template <index_t swizzle_factor = 2>
using WarpGemmMfmaI8I8I32M32N32K32SwizzleBTransposedCDistribution =
    WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
        WarpGemmAttributeMfmaImpl_i32_32x32x16_i8<WGAttrCtlEnum::Default_>,
        2,
        swizzle_factor>>;

#endif // #if USE_NEW_UNIFIED_FRAMEWORK
} // namespace ck_tile
