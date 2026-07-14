// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/scale/scale_traits.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_traits.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @struct amdgcn_mma
 * @brief Specialization for fp8_t scale32 WMMA on GFX1250.
 * @tparam CompilerTarget Current compiler target
 */
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    using ScaleType = int32_t;

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        ScaleType scaleA,
                                        ScaleType scaleB)
    {
        return {__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<fp8_t>,
                                                                 bit_cast<int32x16_t>(aVec),
                                                                 PackedDataTypeToFlag_v<fp8_t>,
                                                                 bit_cast<int32x16_t>(bVec),
                                                                 0, // C_mod
                                                                 cVec,
                                                                 0, // OPSEL[0]
                                                                 0, // Scale Type for A
                                                                 scaleA,
                                                                 0, // OPSEL[1]
                                                                 0, // Scale Type for B
                                                                 scaleB,
                                                                 false,   // matrix_a_reuse
                                                                 false)}; // matrix_b_reuse
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization for bf8_t scale32 WMMA on GFX1250.
 * @tparam CompilerTarget Current compiler target
 */
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    using ScaleType = int32_t;

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        ScaleType scaleA,
                                        ScaleType scaleB)
    {
        return {__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<bf8_t>,
                                                                 bit_cast<int32x16_t>(aVec),
                                                                 PackedDataTypeToFlag_v<bf8_t>,
                                                                 bit_cast<int32x16_t>(bVec),
                                                                 0, // C_mod
                                                                 cVec,
                                                                 0, // OPSEL[0]
                                                                 0, // Scale Type for A
                                                                 scaleA,
                                                                 0, // OPSEL[1]
                                                                 0, // Scale Type for B
                                                                 scaleB,
                                                                 false,   // matrix_a_reuse
                                                                 false)}; // matrix_b_reuse
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization for pk_fp6x16_t scale32 WMMA on GFX1250.
 * @tparam CompilerTarget Current compiler target
 */
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes                    | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    using ScaleType = int32_t;

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        ScaleType scaleA,
                                        ScaleType scaleB)
    {
        int32x16_t a_padded = {aVec.data[0],
                               aVec.data[1],
                               aVec.data[2],
                               aVec.data[3],
                               aVec.data[4],
                               aVec.data[5],
                               aVec.data[6],
                               aVec.data[7],
                               aVec.data[8],
                               aVec.data[9],
                               aVec.data[10],
                               aVec.data[11],
                               0,
                               0,
                               0,
                               0};
        int32x16_t b_padded = {bVec.data[0],
                               bVec.data[1],
                               bVec.data[2],
                               bVec.data[3],
                               bVec.data[4],
                               bVec.data[5],
                               bVec.data[6],
                               bVec.data[7],
                               bVec.data[8],
                               bVec.data[9],
                               bVec.data[10],
                               bVec.data[11],
                               0,
                               0,
                               0,
                               0};
        return {
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<pk_fp6x16_t>,
                                                             a_padded,
                                                             PackedDataTypeToFlag_v<pk_fp6x16_t>,
                                                             b_padded,
                                                             0, // C_mod
                                                             cVec,
                                                             0, // OPSEL[0]
                                                             0, // Scale Type for A
                                                             scaleA,
                                                             0, // OPSEL[1]
                                                             0, // Scale Type for B
                                                             scaleB,
                                                             false,   // matrix_a_reuse
                                                             false)}; // matrix_b_reuse
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization for pk_bf6x16_t scale32 WMMA on GFX1250.
 * @tparam CompilerTarget Current compiler target
 */
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes                    | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    using ScaleType = int32_t;

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        ScaleType scaleA,
                                        ScaleType scaleB)
    {
        int32x16_t a_padded = {aVec.data[0],
                               aVec.data[1],
                               aVec.data[2],
                               aVec.data[3],
                               aVec.data[4],
                               aVec.data[5],
                               aVec.data[6],
                               aVec.data[7],
                               aVec.data[8],
                               aVec.data[9],
                               aVec.data[10],
                               aVec.data[11],
                               0,
                               0,
                               0,
                               0};
        int32x16_t b_padded = {bVec.data[0],
                               bVec.data[1],
                               bVec.data[2],
                               bVec.data[3],
                               bVec.data[4],
                               bVec.data[5],
                               bVec.data[6],
                               bVec.data[7],
                               bVec.data[8],
                               bVec.data[9],
                               bVec.data[10],
                               bVec.data[11],
                               0,
                               0,
                               0,
                               0};
        return {
            __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<pk_bf6x16_t>,
                                                             a_padded,
                                                             PackedDataTypeToFlag_v<pk_bf6x16_t>,
                                                             b_padded,
                                                             0, // C_mod
                                                             cVec,
                                                             0, // OPSEL[0]
                                                             0, // Scale Type for A
                                                             scaleA,
                                                             0, // OPSEL[1]
                                                             0, // Scale Type for B
                                                             scaleB,
                                                             false,   // matrix_a_reuse
                                                             false)}; // matrix_b_reuse
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization for pk_fp4_t scale32 WMMA on GFX1250.
 * @tparam CompilerTarget Current compiler target
 */
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes          | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    using ScaleType = int32_t;

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        ScaleType scaleA,
                                        ScaleType scaleB)
    {
        int32x8_t a8        = bit_cast<int32x8_t>(aVec);
        int32x8_t b8        = bit_cast<int32x8_t>(bVec);
        int32x16_t a_padded = {
            a8[0], a8[1], a8[2], a8[3], a8[4], a8[5], a8[6], a8[7], 0, 0, 0, 0, 0, 0, 0, 0};
        int32x16_t b_padded = {
            b8[0], b8[1], b8[2], b8[3], b8[4], b8[5], b8[6], b8[7], 0, 0, 0, 0, 0, 0, 0, 0};
        return {__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<pk_fp4_t>,
                                                                 a_padded,
                                                                 PackedDataTypeToFlag_v<pk_fp4_t>,
                                                                 b_padded,
                                                                 0, // C_mod
                                                                 cVec,
                                                                 0, // OPSEL[0]
                                                                 0, // Scale Type for A
                                                                 scaleA,
                                                                 0, // OPSEL[1]
                                                                 0, // Scale Type for B
                                                                 scaleB,
                                                                 false,   // matrix_a_reuse
                                                                 false)}; // matrix_b_reuse
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization for fp8_t scale16 WMMA on GFX1250.
 * @tparam CompilerTarget Current compiler target
 */
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE16, enable_if_target_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SCALE16>
// clang-format on
{
    using ScaleType = int64_t;

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        ScaleType scaleA,
                                        ScaleType scaleB)
    {
        return {__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<fp8_t>,
                                                                   bit_cast<int32x16_t>(aVec),
                                                                   PackedDataTypeToFlag_v<fp8_t>,
                                                                   bit_cast<int32x16_t>(bVec),
                                                                   0, // C_mod
                                                                   cVec,
                                                                   0, // OPSEL[0]
                                                                   0, // Scale Type for A
                                                                   scaleA,
                                                                   0, // OPSEL[1]
                                                                   0, // Scale Type for B
                                                                   scaleB,
                                                                   false,   // matrix_a_reuse
                                                                   false)}; // matrix_b_reuse
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization for bf8_t scale16 WMMA on GFX1250.
 * @tparam CompilerTarget Current compiler target
 */
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE16, enable_if_target_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SCALE16>
// clang-format on
{
    using ScaleType = int64_t;

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        ScaleType scaleA,
                                        ScaleType scaleB)
    {
        return {__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<bf8_t>,
                                                                   bit_cast<int32x16_t>(aVec),
                                                                   PackedDataTypeToFlag_v<bf8_t>,
                                                                   bit_cast<int32x16_t>(bVec),
                                                                   0, // C_mod
                                                                   cVec,
                                                                   0, // OPSEL[0]
                                                                   0, // Scale Type for A
                                                                   scaleA,
                                                                   0, // OPSEL[1]
                                                                   0, // Scale Type for B
                                                                   scaleB,
                                                                   false,   // matrix_a_reuse
                                                                   false)}; // matrix_b_reuse
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization for pk_fp6x16_t scale16 WMMA on GFX1250.
 * @tparam CompilerTarget Current compiler target
 */
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes                    | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE16, enable_if_target_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SCALE16>
// clang-format on
{
    using ScaleType = int64_t;

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        ScaleType scaleA,
                                        ScaleType scaleB)
    {
        int32x16_t a_padded = {aVec.data[0],
                               aVec.data[1],
                               aVec.data[2],
                               aVec.data[3],
                               aVec.data[4],
                               aVec.data[5],
                               aVec.data[6],
                               aVec.data[7],
                               aVec.data[8],
                               aVec.data[9],
                               aVec.data[10],
                               aVec.data[11],
                               0,
                               0,
                               0,
                               0};
        int32x16_t b_padded = {bVec.data[0],
                               bVec.data[1],
                               bVec.data[2],
                               bVec.data[3],
                               bVec.data[4],
                               bVec.data[5],
                               bVec.data[6],
                               bVec.data[7],
                               bVec.data[8],
                               bVec.data[9],
                               bVec.data[10],
                               bVec.data[11],
                               0,
                               0,
                               0,
                               0};
        return {
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<pk_fp6x16_t>,
                                                               a_padded,
                                                               PackedDataTypeToFlag_v<pk_fp6x16_t>,
                                                               b_padded,
                                                               0, // C_mod
                                                               cVec,
                                                               0, // OPSEL[0]
                                                               0, // Scale Type for A
                                                               scaleA,
                                                               0, // OPSEL[1]
                                                               0, // Scale Type for B
                                                               scaleB,
                                                               false,   // matrix_a_reuse
                                                               false)}; // matrix_b_reuse
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization for pk_bf6x16_t scale16 WMMA on GFX1250.
 * @tparam CompilerTarget Current compiler target
 */
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes                    | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE16, enable_if_target_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SCALE16>
// clang-format on
{
    using ScaleType = int64_t;

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        ScaleType scaleA,
                                        ScaleType scaleB)
    {
        int32x16_t a_padded = {aVec.data[0],
                               aVec.data[1],
                               aVec.data[2],
                               aVec.data[3],
                               aVec.data[4],
                               aVec.data[5],
                               aVec.data[6],
                               aVec.data[7],
                               aVec.data[8],
                               aVec.data[9],
                               aVec.data[10],
                               aVec.data[11],
                               0,
                               0,
                               0,
                               0};
        int32x16_t b_padded = {bVec.data[0],
                               bVec.data[1],
                               bVec.data[2],
                               bVec.data[3],
                               bVec.data[4],
                               bVec.data[5],
                               bVec.data[6],
                               bVec.data[7],
                               bVec.data[8],
                               bVec.data[9],
                               bVec.data[10],
                               bVec.data[11],
                               0,
                               0,
                               0,
                               0};
        return {
            __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<pk_bf6x16_t>,
                                                               a_padded,
                                                               PackedDataTypeToFlag_v<pk_bf6x16_t>,
                                                               b_padded,
                                                               0, // C_mod
                                                               cVec,
                                                               0, // OPSEL[0]
                                                               0, // Scale Type for A
                                                               scaleA,
                                                               0, // OPSEL[1]
                                                               0, // Scale Type for B
                                                               scaleB,
                                                               false,   // matrix_a_reuse
                                                               false)}; // matrix_b_reuse
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization for pk_fp4_t scale16 WMMA on GFX1250.
 * @tparam CompilerTarget Current compiler target
 */
template <typename CompilerTarget>
// clang-format off
//               | A B C DataTypes          | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE16, enable_if_target_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SCALE16>
// clang-format on
{
    using ScaleType = int64_t;

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        ScaleType scaleA,
                                        ScaleType scaleB)
    {
        int32x8_t a8        = bit_cast<int32x8_t>(aVec);
        int32x8_t b8        = bit_cast<int32x8_t>(bVec);
        int32x16_t a_padded = {
            a8[0], a8[1], a8[2], a8[3], a8[4], a8[5], a8[6], a8[7], 0, 0, 0, 0, 0, 0, 0, 0};
        int32x16_t b_padded = {
            b8[0], b8[1], b8[2], b8[3], b8[4], b8[5], b8[6], b8[7], 0, 0, 0, 0, 0, 0, 0, 0};
        return {__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<pk_fp4_t>,
                                                                   a_padded,
                                                                   PackedDataTypeToFlag_v<pk_fp4_t>,
                                                                   b_padded,
                                                                   0, // C_mod
                                                                   cVec,
                                                                   0, // OPSEL[0]
                                                                   0, // Scale Type for A
                                                                   scaleA,
                                                                   0, // OPSEL[1]
                                                                   0, // Scale Type for B
                                                                   scaleB,
                                                                   false,   // matrix_a_reuse
                                                                   false)}; // matrix_b_reuse
    }
};

} // namespace ck_tile::core::arch::mma
