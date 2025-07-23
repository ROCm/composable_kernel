// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

namespace impl {
template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC,
          bool SwizzleA                     = false,
          bool UseStructuredSparsity        = false,
          WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
struct WarpGemmMfmaDispatcher;

// clang-format off
// fp16
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32,  8, false> { using Type = WarpGemmMfmaF16F16F32M32N32K8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32,  8, true> { using Type = WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false> { using Type = WarpGemmMfmaF16F16F32M32N32K16<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, true> { using Type = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false, false, false, WGAttrNumAccessEnum::Double> {
    using Type = WarpGemmMfmaF16F16F32M32N32K16<WGAttrNumAccessEnum::Double>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, true, false, false, WGAttrNumAccessEnum::Double> {
    using Type = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution<WGAttrNumAccessEnum::Double>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 16, false> { using Type = WarpGemmMfmaF16F16F32M16N16K16; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 16, true> { using Type = WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, false> { using Type = WarpGemmMfmaF16F16F32M16N16K32<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, true> { using Type = WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, false, false, false, WGAttrNumAccessEnum::Double> {
    using Type = WarpGemmMfmaF16F16F32M16N16K32<WGAttrNumAccessEnum::Double>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, true, false, false, WGAttrNumAccessEnum::Double> {
    using Type = WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution<WGAttrNumAccessEnum::Double>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 4, 64, 16, false> { using Type = WarpGemmMfmaF16F16F32M4N64K16; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 64, 4, 16, false> { using Type = WarpGemmMfmaF16F16F32M64N4K16; };

template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32,  8, false, true> { using Type = WarpGemmMfmaF16F16F32M32N32K8SwizzleA; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false, true> { using Type = WarpGemmMfmaF16F16F32M32N32K16SwizzleA; };

// fp16 2:4 structural sparsity
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false, false, true> { using Type = WarpGemmSmfmacF16F16F32M32N32K16; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, false, false, true> { using Type = WarpGemmSmfmacF16F16F32M16N16K32; };

// bf16
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32,  8, false> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32,  8, true> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, false> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, true> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, false, false, false, WGAttrNumAccessEnum::Double> {
    using Type = WarpGemmMfmaBf16Bf16F32M32N32K16<WGAttrNumAccessEnum::Double>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, true, false, false, WGAttrNumAccessEnum::Double> {
    using Type = WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution<WGAttrNumAccessEnum::Double>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 16, false> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K16; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 16, true> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, false> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K32<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, true> { using Type = WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, false, false, false, WGAttrNumAccessEnum::Double> {
    using Type = WarpGemmMfmaBf16Bf16F32M16N16K32<WGAttrNumAccessEnum::Double>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, true, false, false, WGAttrNumAccessEnum::Double> {
    using Type = WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution<WGAttrNumAccessEnum::Double>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 4, 64, 16, false> { using Type = WarpGemmMfmaBf16Bf16F32M4N64K16; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 64, 4, 16, false> { using Type = WarpGemmMfmaBf16Bf16F32M64N4K16; };

template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32,  8, false, true> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K8SwizzleA; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, false, true> { using Type = WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleA; };

// fp8
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_fp8_fp8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32,  32, false> { using Type = WarpGemmMfma_f32_32x32x32_fp8_fp8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 16, 16,  32, false> { using Type = WarpGemmMfma_f32_16x16x32_fp8_fp8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 16, 16,  64, false> { using Type = WarpGemmMfma_f32_16x16x64_fp8_fp8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32,  16, true> { using Type = WarpGemmMfma_f32_32x32x16_fp8_fp8_CTransposed; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_fp8_bf8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32,  16, true> { using Type = WarpGemmMfma_f32_32x32x16_fp8_bf8_CTransposed; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_bf8_fp8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32,  16, true> { using Type = WarpGemmMfma_f32_32x32x16_bf8_fp8_CTransposed; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32,  16, false> { using Type = WarpGemmMfma_f32_32x32x16_bf8_bf8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32,  32, false> { using Type = WarpGemmMfma_f32_32x32x32_bf8_bf8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 16, 16,  32, false> { using Type = WarpGemmMfma_f32_16x16x32_bf8_bf8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 16, 16,  64, false> { using Type = WarpGemmMfma_f32_16x16x64_bf8_bf8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32,  16, true> { using Type = WarpGemmMfma_f32_32x32x16_bf8_bf8_CTransposed; };

template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 16, 16,  128, false> { using Type = WarpGemmMfma_f32_16x16x128_fp8_fp8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::bf8_t, float, 16, 16,  128, false> { using Type = WarpGemmMfma_f32_16x16x128_fp8_bf8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::fp8_t, float, 16, 16,  128, false> { using Type = WarpGemmMfma_f32_16x16x128_bf8_fp8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 16, 16,  128, false> { using Type = WarpGemmMfma_f32_16x16x128_bf8_bf8; };

template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32,  64, false> { using Type = WarpGemmMfma_f32_32x32x64_fp8_fp8<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32,  64, false> { using Type = WarpGemmMfma_f32_32x32x64_fp8_bf8<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32,  64, false> { using Type = WarpGemmMfma_f32_32x32x64_bf8_fp8<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32,  64, false> { using Type = WarpGemmMfma_f32_32x32x64_bf8_bf8<>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32,  64, false, false, false, WGAttrNumAccessEnum::Quad> {
    using Type = WarpGemmMfma_f32_32x32x64_fp8_fp8<WGAttrNumAccessEnum::Quad>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::fp8_t, ck_tile::bf8_t, float, 32, 32,  64, false, false, false, WGAttrNumAccessEnum::Quad> {
    using Type = WarpGemmMfma_f32_32x32x64_fp8_bf8<WGAttrNumAccessEnum::Quad>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::fp8_t, float, 32, 32,  64, false, false, false, WGAttrNumAccessEnum::Quad> {
    using Type = WarpGemmMfma_f32_32x32x64_bf8_fp8<WGAttrNumAccessEnum::Quad>; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32,  64, false, false, false, WGAttrNumAccessEnum::Quad> {
    using Type = WarpGemmMfma_f32_32x32x64_bf8_bf8<WGAttrNumAccessEnum::Quad>; };

// int8
// ADataType, BDataType, AccDataType, MPerWave, NPerWave, KPerWave, TransposeC, SwizzleA, UseStructuredSparsity
template<> struct WarpGemmMfmaDispatcher<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 32, 32,  16, false> { using Type = WarpGemmMfma_i32_32x32x16_i8_i8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 32, 32,  16, true> { using Type = WarpGemmMfma_i32_32x32x16_i8_i8_CTransposed; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 16, 16,  32, false> { using Type = WarpGemmMfma_i32_16x16x32_i8_i8; };
template<> struct WarpGemmMfmaDispatcher<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 16, 16,  32, true> { using Type = WarpGemmMfma_i32_16x16x32_i8_i8_CTransposed; };

// clang-format on
} // namespace impl

template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC,
          bool SwizzleA                     = false,
          bool UseStructuredSparsity        = false,
          WGAttrNumAccessEnum AttrNumAccess = WGAttrNumAccessEnum::Single>
using WarpGemmMfmaDispatcher = typename impl::WarpGemmMfmaDispatcher<AType,
                                                                     BType,
                                                                     AccType,
                                                                     MPerWave,
                                                                     NPerWave,
                                                                     KPerWave,
                                                                     TransposeC,
                                                                     SwizzleA,
                                                                     UseStructuredSparsity,
                                                                     AttrNumAccess>::Type;

} // namespace ck_tile
