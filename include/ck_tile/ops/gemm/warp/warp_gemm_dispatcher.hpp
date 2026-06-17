// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_wmma_gemm.hpp"

namespace ck_tile {

namespace impl {
namespace warp_gemm_dispatcher {

// C++20 using enum
static inline constexpr auto ESingle  = WGAttrNumAccessEnum::Single;
static inline constexpr auto EDouble  = WGAttrNumAccessEnum::Double;
static inline constexpr auto EQuad    = WGAttrNumAccessEnum::Quad;
static inline constexpr auto EDefault = WGAttrNumAccessEnum::Default;

struct WmmaTag
{
};

template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC,
          bool SwizzleA                      = false,
          bool UseStructuredSparsity         = false,
          WGAttrNumAccessEnum AttrNumAccessA = ESingle,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA,
          bool IsScale16                     = false,
          typename Enable                    = void>
struct Dispatcher;

// clang-format off
// fp32
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct Dispatcher<float, float, float, 16, 16, 16, false> { using Type = WarpGemmMfmaF32F32F32M16N16K16<>; };
template<> struct Dispatcher<float, float, float, 16, 16,  8, false> { using Type = WarpGemmMfmaF32F32F32M16N16K8<>; };
template<> struct Dispatcher<float, float, float, 32, 32,  4, false> { using Type = WarpGemmMfmaF32F32F32M32N32K4<>; };
template<> struct Dispatcher<float, float, float, 32, 32,  8, false> { using Type = WarpGemmMfmaF32F32F32M32N32K8<>; };
template<> struct Dispatcher<float, float, float, 32, 32,  8, false, false, false, EDouble> { using Type = WarpGemmMfmaF32F32F32M32N32K8<EDouble>; };
template<> struct Dispatcher<float, float, float, 16, 16, 16,  true> { using Type = WarpGemmMfmaF32F32F32M16N16K16TransposedCDistribution<>; };

// tf32 (on gfx950: uses 3x bf16 MFMA emulation)
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
#if defined(CK_GFX950_SUPPORT)
template<> struct Dispatcher<tf32_t, tf32_t, float, 32, 32, 16, false> { using Type = WarpGemmMfmaTf32Tf32F32M32N32K16<>; };
template<> struct Dispatcher<tf32_t, tf32_t, float, 32, 32, 16,  true> { using Type = WarpGemmMfmaTf32Tf32F32M32N32K16<>; };
template<> struct Dispatcher<tf32_t, tf32_t, float, 32, 32, 16, false, false, false, EDouble> { using Type = WarpGemmMfmaTf32Tf32F32M32N32K16<EDouble>; };
template<> struct Dispatcher<tf32_t, tf32_t, float, 32, 32, 16, false, false, false, EQuad> { using Type = WarpGemmMfmaTf32Tf32F32M32N32K16<EQuad>; };
// TF32 16x16x32 for weight preshuffle pipeline (uses native 16x16x32 TF32 MFMA emulation)
template<> struct Dispatcher<tf32_t, tf32_t, float, 16, 16, 32, false> { using Type = WarpGemmMfmaTf32Tf32F32M16N16K32<>; };
template<> struct Dispatcher<tf32_t, tf32_t, float, 16, 16, 32, false, false, false, EDouble> { using Type = WarpGemmMfmaTf32Tf32F32M16N16K32<EDouble>; };
template<> struct Dispatcher<tf32_t, tf32_t, float, 16, 16, 32, false, false, false, EQuad> { using Type = WarpGemmMfmaTf32Tf32F32M16N16K32<EQuad>; };
#endif
// Note: For gfx11/gfx12 and other architectures that don't support tf32,
// these dispatchers are not defined. Code using tf32 should be guarded
// by CK_ENABLE_TF32 or CK_GFX950_SUPPORT macros.
// WMMA cases
#if defined(__gfx125__)
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess>
struct Dispatcher<float, float, float, 16, 16, 4, TransposeC, false, false, AttrNumAccess, AttrNumAccess> 
    : WmmaTag { using Type = WarpGemmWmma_f32_16x16x4_f32<TransposeC, AttrNumAccess>;};
#else
template<> struct Dispatcher<float, float, float, 16, 16, 4, false> { using Type = WarpGemmMfmaF32F32F32M16N16K4; };
template<> struct Dispatcher<float, float, float, 16, 16, 4, true> { using Type = WarpGemmWmma_f32_16x16x4_f32<true>; };
#endif
// fp16
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct Dispatcher<half_t, half_t, float, 32, 32,  8, false> { using Type = WarpGemmMfmaF16F16F32M32N32K8; };
template<> struct Dispatcher<half_t, half_t, float, 32, 32,  8,  true>  { using Type = WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution; };
template<> struct Dispatcher<half_t, half_t, float, 32, 32, 16, false> { using Type = WarpGemmMfmaF16F16F32M32N32K16<>; };
template<> struct Dispatcher<half_t, half_t, float, 32, 32, 16,  true>  { using Type = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution<>; };
template<> struct Dispatcher<half_t, half_t, float, 32, 32, 16, false, false, false, EDouble> { using Type = WarpGemmMfmaF16F16F32M32N32K16<EDouble>; };
template<> struct Dispatcher<half_t, half_t, float, 32, 32, 16,  true, false, false, EDouble> { using Type = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution<EDouble>; };
#if defined(__gfx125__)
template<bool TransposeC> struct Dispatcher<half_t, half_t, float, 16, 16, 32, TransposeC, false, false, EDouble> : WmmaTag 
       { using Type = WarpGemmWmma_f32_16x16x32_f16_f16<TransposeC, EDouble>;};
#else
template<> struct Dispatcher<half_t, half_t, float, 16, 16, 32, false, false, false, EDouble> { using Type = WarpGemmMfmaF16F16F32M16N16K32<EDouble>; };
template<> struct Dispatcher<half_t, half_t, float, 16, 16, 32,  true, false, false, EDouble> { using Type = WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution<EDouble>; };
#endif // defined(__gfx125__)
template<> struct Dispatcher<half_t, half_t, float,  4, 64, 16, false> { using Type = WarpGemmMfmaF16F16F32M4N64K16; };
template<> struct Dispatcher<half_t, half_t, float, 64,  4, 16, false> { using Type = WarpGemmMfmaF16F16F32M64N4K16; };
// WMMA cases
#if defined(__gfx11__) || defined(__gfx120__)
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<half_t, half_t, float, 16, 16, 16, TransposeC, false, false, AttrNumAccess, AttrNumAccess> 
    : WmmaTag { using Type = WarpGemmWmma_f32_16x16x16_f16_f16<TransposeC, AttrNumAccess>;};
#else
template<> struct Dispatcher<half_t, half_t, float, 16, 16, 16, false> { using Type = WarpGemmMfmaF16F16F32M16N16K16; };
template<> struct Dispatcher<half_t, half_t, float, 16, 16, 16,  true>  { using Type = WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution; };
#endif

#if defined(__gfx125__)
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<half_t, half_t, float, 16, 16, 32, TransposeC, false, false, AttrNumAccess, AttrNumAccess> 
    : WmmaTag { using Type = WarpGemmWmma_f32_16x16x32_f16_f16<TransposeC, AttrNumAccess>;};
#else
template<> struct Dispatcher<half_t, half_t, float, 16, 16, 32, false> { using Type = WarpGemmMfmaF16F16F32M16N16K32<>; };
template<> struct Dispatcher<half_t, half_t, float, 16, 16, 32, true>  { using Type = WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution<>; };
#endif

template<> struct Dispatcher<half_t, half_t, float, 32, 32,  8, false, true> { using Type = WarpGemmMfmaF16F16F32M32N32K8SwizzleA; };
template<> struct Dispatcher<half_t, half_t, float, 32, 32, 16, false, true> { using Type = WarpGemmMfmaF16F16F32M32N32K16SwizzleA; };
template<> struct Dispatcher<half_t, half_t, float, 32, 32,  8,  true, true> { using Type = WarpGemmMfmaF16F16F32M32N32K8SwizzleBTransposedCDistribution; };
template<> struct Dispatcher<half_t, half_t, float, 32, 32, 16,  true, true> { using Type = WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution; };


// fp16 2:4 structural sparsity
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct Dispatcher<half_t, half_t, float, 32, 32, 16, false, false, true> { using Type = WarpGemmSmfmacF16F16F32M32N32K16; };
template<> struct Dispatcher<half_t, half_t, float, 16, 16, 32, false, false, true> { using Type = WarpGemmSmfmacF16F16F32M16N16K32; };

// bf16
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct Dispatcher<bf16_t, bf16_t, float, 32, 32,  8, false> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K8; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 32, 32,  8,  true>  { using Type = WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 32, 32, 16, false> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16<>; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 32, 32, 16,  true>  { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution<>; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 32, 32, 16, false, false, false, EDouble> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16<EDouble>; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 32, 32, 16,  true, false, false, EDouble> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution<EDouble>; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 32, false, false, false, EDouble, ESingle> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K32<EDouble, ESingle>; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 64, false, false, false, EQuad, ESingle> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K64<EQuad, ESingle>; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 64, false, false, false, EQuad> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K64<EQuad>; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 64, false> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K64<>; };
#if defined(__gfx125__)
template<bool TransposeC> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 32, TransposeC, false, false, EDouble> : WmmaTag 
    { using Type = WarpGemmWmma_f32_16x16x32_bf16_bf16<TransposeC, EDouble>;};
#else
template<> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 32, false, false, false, EDouble> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K32<EDouble>; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 32,  true, false, false, EDouble> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution<EDouble>; };
#endif // defined(__gfx125__)
template<> struct Dispatcher<bf16_t, bf16_t, float,  4, 64, 16, false> { using Type = WarpGemmMfmaBf16Bf16F32M4N64K16; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 64,  4, 16, false> { using Type = WarpGemmMfmaBf16Bf16F32M64N4K16; };
// WMMA cases
#if defined(__gfx11__) || defined(__gfx120__)
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 16, TransposeC, false, false, AttrNumAccess, AttrNumAccess>
    : WmmaTag { using Type = WarpGemmWmma_f32_16x16x16_bf16_bf16<TransposeC, AttrNumAccess>; };
#else
template<> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 16, false> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K16; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 16,  true> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution; };
#endif

#if defined(__gfx125__)
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 32, TransposeC, false, false, AttrNumAccess, AttrNumAccess>
    : WmmaTag { using Type = WarpGemmWmma_f32_16x16x32_bf16_bf16<TransposeC, AttrNumAccess>;};
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf16_t, bf16_t, bf16_t, 16, 16, 32, TransposeC, false, false, AttrNumAccess, AttrNumAccess>
    : WmmaTag { using Type = WarpGemmWmma_bf16_16x16x32_bf16_bf16<TransposeC, AttrNumAccess>;};
#else
template<> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 32, false> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K32<>; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 16, 16, 32, true>  { using Type = WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution<>; };
#endif

template<> struct Dispatcher<bf16_t, bf16_t, float, 32, 32,  8, false, true> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K8SwizzleA; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 32, 32, 16, false, true> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 32, 32,  8,  true, true> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K8SwizzleBTransposedCDistribution; };
template<> struct Dispatcher<bf16_t, bf16_t, float, 32, 32, 16,  true, true> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution; };

// fp8
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct Dispatcher<fp8_t, fp8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_fp8_fp8; };
template<> struct Dispatcher<fp8_t, fp8_t, float, 16, 16,  32, false> { using Type = WarpGemmMfma_f32_16x16x32_fp8_fp8; };
template<> struct Dispatcher<fp8_t, fp8_t, float, 32, 32,  16,  true> { using Type = WarpGemmMfma_f32_32x32x16_fp8_fp8_CTransposed; };
template<> struct Dispatcher<fp8_t, fp8_t, float, 16, 16,  32,  true> { using Type = WarpGemmMfma_f32_16x16x32_fp8_fp8_CTransposed; };
template<> struct Dispatcher<fp8_t, bf8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_fp8_bf8; };
template<> struct Dispatcher<fp8_t, bf8_t, float, 32, 32,  16,  true> { using Type = WarpGemmMfma_f32_32x32x16_fp8_bf8_CTransposed; };
template<> struct Dispatcher<fp8_t, bf8_t, float, 16, 16,  32, false> { using Type = WarpGemmMfma_f32_16x16x32_fp8_bf8; };
template<> struct Dispatcher<bf8_t, fp8_t, float, 16, 16,  32, false> { using Type = WarpGemmMfma_f32_16x16x32_bf8_fp8; };
template<> struct Dispatcher<fp8_t, bf8_t, float, 32, 32,  32, false> { using Type = WarpGemmMfma_f32_32x32x32_fp8_bf8; };
template<> struct Dispatcher<bf8_t, fp8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_bf8_fp8; };
template<> struct Dispatcher<bf8_t, fp8_t, float, 32, 32,  16,  true> { using Type = WarpGemmMfma_f32_32x32x16_bf8_fp8_CTransposed; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_bf8_bf8; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 16, 16,  32, false> { using Type = WarpGemmMfma_f32_16x16x32_bf8_bf8; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 16, 16,  32,  true> { using Type = WarpGemmMfma_f32_16x16x32_bf8_bf8_CTransposed; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 32, 32,  16,  true> { using Type = WarpGemmMfma_f32_32x32x16_bf8_bf8_CTransposed; };

#if !defined(__gfx125__)
// scale mfma based f8f6f4
template<typename A, typename B, WGAttrNumAccessEnum I, bool IsScale16>
struct Dispatcher<A, B, float, 16, 16, 128, false, false, false, I, I, IsScale16, std::enable_if_t<I != EDefault>> { using Type = WarpGemmMfma_f32_16x16x128_f8f6f4<A, B, I>; };
template<typename A, typename B, WGAttrNumAccessEnum I, bool IsScale16>
struct Dispatcher<A, B, float, 16, 16, 128, true, false, false, I, I, IsScale16, std::enable_if_t<I != EDefault>> { using Type = WarpGemmMfma_f32_16x16x128_f8f6f4_CTransposed<A, B, I>; };
#endif

template<> struct Dispatcher<fp8_t, fp8_t, float, 32, 32,  64, false> { using Type = WarpGemmMfma_f32_32x32x64_fp8_fp8<>; };
template<> struct Dispatcher<fp8_t, bf8_t, float, 32, 32,  64, false> { using Type = WarpGemmMfma_f32_32x32x64_fp8_bf8<>; };
template<> struct Dispatcher<bf8_t, fp8_t, float, 32, 32,  64, false> { using Type = WarpGemmMfma_f32_32x32x64_bf8_fp8<>; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 32, 32,  64, false> { using Type = WarpGemmMfma_f32_32x32x64_bf8_bf8<>; };
template<> struct Dispatcher<fp8_t, fp8_t, float, 32, 32,  64, false, false, false, EDouble> { using Type = WarpGemmMfma_f32_32x32x64_fp8_fp8<EDouble>; };
template<> struct Dispatcher<fp8_t, bf8_t, float, 32, 32,  64, false, false, false, EDouble> { using Type = WarpGemmMfma_f32_32x32x64_fp8_bf8<EDouble>; };
template<> struct Dispatcher<bf8_t, fp8_t, float, 32, 32,  64, false, false, false, EDouble> { using Type = WarpGemmMfma_f32_32x32x64_bf8_fp8<EDouble>; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 32, 32,  64, false, false, false, EDouble> { using Type = WarpGemmMfma_f32_32x32x64_bf8_bf8<EDouble>; };
template<> struct Dispatcher<fp8_t, fp8_t, float, 32, 32,  64, false, false, false, EQuad> { using Type = WarpGemmMfma_f32_32x32x64_fp8_fp8<EQuad>; };
template<> struct Dispatcher<fp8_t, bf8_t, float, 32, 32,  64, false, false, false, EQuad> { using Type = WarpGemmMfma_f32_32x32x64_fp8_bf8<EQuad>; };
template<> struct Dispatcher<bf8_t, fp8_t, float, 32, 32,  64, false, false, false, EQuad> { using Type = WarpGemmMfma_f32_32x32x64_bf8_fp8<EQuad>; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 32, 32,  64, false, false, false, EQuad> { using Type = WarpGemmMfma_f32_32x32x64_bf8_bf8<EQuad>; };

template<WGAttrNumAccessEnum I> struct Dispatcher<fp8_t, fp8_t, float, 32, 32,  64,  true, false, false, I> { using Type = WarpGemmMfma_f32_32x32x64_fp8_fp8_CTransposed<I>; };
template<WGAttrNumAccessEnum I> struct Dispatcher<fp8_t, bf8_t, float, 32, 32,  64,  true, false, false, I> { using Type = WarpGemmMfma_f32_32x32x64_fp8_bf8_CTransposed<I>; };
template<WGAttrNumAccessEnum I> struct Dispatcher<bf8_t, fp8_t, float, 32, 32,  64,  true, false, false, I> { using Type = WarpGemmMfma_f32_32x32x64_bf8_fp8_CTransposed<I>; };
template<WGAttrNumAccessEnum I> struct Dispatcher<bf8_t, bf8_t, float, 32, 32,  64,  true, false, false, I> { using Type = WarpGemmMfma_f32_32x32x64_bf8_bf8_CTransposed<I>; };

template<WGAttrNumAccessEnum I> struct Dispatcher<pk_fp4_t, pk_fp4_t, float, 32, 32,  64,  true, false, false, I> { using Type = WarpGemmMfma_f32_32x32x64_fp4_fp4_CTransposed<I>; };

template<WGAttrNumAccessEnum I> struct Dispatcher<pk_fp4_t, pk_fp6x16_t, float, 32, 32,  64,  true, false, false, I> { using Type = WarpGemmMfma_f32_32x32x64_fp4_fp6_CTransposed<I>; };

template<> struct Dispatcher<fp8_t, fp8_t, float, 32, 32,  32, false> { using Type = WarpGemmMfma_f32_32x32x32_fp8_fp8<>; };
template<> struct Dispatcher<fp8_t, fp8_t, float, 32, 32,  32, false, false, false, EDouble> { using Type = WarpGemmMfma_f32_32x32x32_fp8_fp8<EDouble>; };
template<> struct Dispatcher<fp8_t, fp8_t, float, 32, 32,  32, true, false, false> { using Type = WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>; };
template<> struct Dispatcher<fp8_t, fp8_t, float, 32, 32,  32, true, false, false, EDouble> { using Type = WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<EDouble>; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 32, 32,  32, false> { using Type = WarpGemmMfma_f32_32x32x32_bf8_bf8<>; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 32, 32,  32, false, false, false, EDouble> { using Type = WarpGemmMfma_f32_32x32x32_bf8_bf8<EDouble>; };

template<> struct Dispatcher<fp8_t, fp8_t, float, 16, 16,  64, false, false, false, EDouble> { using Type = WarpGemmMfma_f32_16x16x64_fp8_fp8<EDouble>; };

//WMMA cases
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<fp8_t, fp8_t, float, 16, 16, 16, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x16_f8_f8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf8_t, bf8_t, float, 16, 16, 16, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x16_bf8_bf8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<fp8_t, bf8_t, float, 16, 16, 16, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x16_f8_bf8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf8_t, fp8_t, float, 16, 16, 16, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x16_bf8_f8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<fp8_t, bf8_t, float, 16, 16, 64, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x64_f8_bf8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf8_t, fp8_t, float, 16, 16, 64, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x64_bf8_f8<TransposeC, AttrNumAccess>; };

template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<pk_fp4_t, pk_fp4_t, float, 32, 16, 128, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_32x16x128_f4<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess, bool IsScale16> struct Dispatcher<pk_fp4_t, pk_fp4_t, float, 32, 32, 128, TransposeC, false, false, AttrNumAccess, AttrNumAccess, IsScale16> : WmmaTag { using Type = WarpGemmWmma_f32_32x32x128_f4<TransposeC, AttrNumAccess, IsScale16>; };

#if defined(__gfx125__)
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<fp8_t, fp8_t, float, 16, 16,  64, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x64_f8_f8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf8_t, bf8_t, float, 16, 16,  64, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x64_bf8_bf8<TransposeC, AttrNumAccess>; };

template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<fp8_t, fp8_t, float, 16, 16,  128, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x128_f8_f8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf8_t, bf8_t, float, 16, 16,  128, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x128_bf8_bf8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<fp8_t, bf8_t, float, 16, 16,  128, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x128_f8_bf8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf8_t, fp8_t, float, 16, 16,  128, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x128_bf8_f8<TransposeC, AttrNumAccess>; };

// F8F6F4 Mixed precision cases
template<typename A, typename B, bool TransposeC, WGAttrNumAccessEnum AttrNumAccessA, WGAttrNumAccessEnum AttrNumAccessB> struct Dispatcher<A, B, float, 16, 16, 128, TransposeC, false, false, AttrNumAccessA, AttrNumAccessB> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x128_f8f6f4<A, B, TransposeC, AttrNumAccessA, AttrNumAccessB>; };

// F8F6F4 Scale16 (IsScale16=true)
template<typename A, typename B, bool TransposeC, WGAttrNumAccessEnum AttrNumAccessA, WGAttrNumAccessEnum AttrNumAccessB> struct Dispatcher<A, B, float, 16, 16, 128, TransposeC, false, false, AttrNumAccessA, AttrNumAccessB, true> : WmmaTag { using Type = WarpGemmWmma_f32_16x16x128_f8f6f4_scale16<A, B, TransposeC, AttrNumAccessA, AttrNumAccessB>; };
#else
template<> struct Dispatcher<fp8_t, fp8_t, float, 16, 16,  64, false> { using Type = WarpGemmMfma_f32_16x16x64_fp8_fp8<>; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 16, 16,  64, false> { using Type = WarpGemmMfma_f32_16x16x64_bf8_bf8; };
template<> struct Dispatcher<fp8_t, fp8_t, float, 16, 16,  64, true> { using Type = WarpGemmMfma_f32_16x16x64_fp8_fp8_CTransposed; };
template<> struct Dispatcher<bf8_t, bf8_t, float, 16, 16,  64, true> { using Type = WarpGemmMfma_f32_16x16x64_bf8_bf8_CTransposed; };
#endif

template<typename A,
         typename B,
         bool TransposeC,
         WGAttrNumAccessEnum AttrNumAccessA,
         WGAttrNumAccessEnum AttrNumAccessB,
         bool IsScale16>
struct Dispatcher<A,
                  B,
                  float,
                  32,
                  32,
                  128,
                  TransposeC,
                  false,
                  false,
                  AttrNumAccessA,
                  AttrNumAccessB,
                  IsScale16> : WmmaTag
{
    using Type =
        std::conditional_t<IsScale16,
                           WarpGemmWmma_f32_32x32x128_f8f6f4_scale16<A, B, TransposeC, AttrNumAccessA, AttrNumAccessB>,
                           WarpGemmWmma_f32_32x32x128_f8f6f4<A, B, TransposeC, AttrNumAccessA, AttrNumAccessB>>;
};

template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<fp8_t, fp8_t, half_t, 16, 16,  64, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type =WarpGemmWmma_f16_16x16x64_f8_f8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf8_t, bf8_t, half_t, 16, 16,  64, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type =WarpGemmWmma_f16_16x16x64_bf8_bf8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<fp8_t, bf8_t, half_t, 16, 16,  64, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type =WarpGemmWmma_f16_16x16x64_f8_bf8<TransposeC, AttrNumAccess>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<bf8_t, fp8_t, half_t, 16, 16,  64, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type =WarpGemmWmma_f16_16x16x64_bf8_f8<TransposeC, AttrNumAccess>; };

// int8
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct Dispatcher<int8_t, int8_t, int32_t, 32, 32, 16, false> { using Type = WarpGemmMfma_i32_32x32x16_i8_i8; };
template<> struct Dispatcher<int8_t, int8_t, int32_t, 32, 32, 16,  true> { using Type = WarpGemmMfma_i32_32x32x16_i8_i8_CTransposed; };
template<> struct Dispatcher<int8_t, int8_t, int32_t, 16, 16, 32, false> { using Type = WarpGemmMfma_i32_16x16x32_i8_i8; };
template<> struct Dispatcher<int8_t, int8_t, int32_t, 16, 16, 32,  true> { using Type = WarpGemmMfma_i32_16x16x32_i8_i8_CTransposed; };
// WMMA cases
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<int8_t, int8_t, int32_t, 16, 16, 16, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_i32_16x16x16_i8_i8<TransposeC>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<int8_t, int8_t, int32_t, 16, 16, 64, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_i32_16x16x64_i8_i8<TransposeC>; };
template<bool TransposeC, WGAttrNumAccessEnum AttrNumAccess> struct Dispatcher<uint8_t, uint8_t, int32_t, 16, 16, 64, TransposeC, false, false, AttrNumAccess, AttrNumAccess> : WmmaTag { using Type = WarpGemmWmma_i32_16x16x64_u8_u8<TransposeC>; };

template <typename AType, typename BType, typename AccType,
          index_t M, index_t N, index_t K,
          bool TransposeC, bool SA, bool SS, bool IsScale16>
struct Dispatcher<AType, BType, AccType, M, N, K, TransposeC, SA, SS,
                  EDefault, EDefault, IsScale16,
                  std::enable_if_t<!std::is_base_of_v<WmmaTag,
                      Dispatcher<AType, BType, AccType, M, N, K, TransposeC, SA, SS, ESingle, ESingle, IsScale16, void>>>>
    : Dispatcher<AType, BType, AccType, M, N, K, TransposeC, SA, SS, ESingle, ESingle, IsScale16, void> {};

// clang-format on
} // namespace warp_gemm_dispatcher
} // namespace impl

template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC,
          bool SwizzleA                      = false,
          bool UseStructuredSparsity         = false,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Default,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA,
          bool IsScale16                     = false>
using WarpGemmDispatcher = typename impl::warp_gemm_dispatcher::Dispatcher< //
    AType,
    BType,
    AccType,
    MPerWave,
    NPerWave,
    KPerWave,
    TransposeC,
    SwizzleA,
    UseStructuredSparsity,
    AttrNumAccessA,
    AttrNumAccessB,
    IsScale16>::Type;

} // namespace ck_tile
