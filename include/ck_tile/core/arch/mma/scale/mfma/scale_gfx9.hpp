// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @defgroup scale_mfma_gfx9 Scale MFMA for GFX9
 * @brief Scale specializations of @ref amdgcn_mma for GFX9 family.
 *
 * Template parameters A/B/C denote input/output types,
 * M/N/K are the fragment (MmaTile) sizes,
 * and `enable_if_target_*` restricts the specialization to specific GPU targets.
 *
 * @tparam CompilerTarget Current compiler target.
 *
 * @sa amdgcn_mma_base for base template parameter documentation.
 * @{
 */

// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires

// clang-format off
#define MMA_SCALE_ARG_F8(vec) bit_cast<int32x8_t>(vec)
#define MMA_SCALE_ARG_F6(vec) int32x8_t{vec.data[0], vec.data[1], vec.data[2], vec.data[3], vec.data[4], vec.data[5], 0, 0}
#define MMA_SCALE_ARG_F4(vec) int32x8_t{bit_cast<int32x4_t>(vec)[0], bit_cast<int32x4_t>(vec)[1], bit_cast<int32x4_t>(vec)[2], bit_cast<int32x4_t>(vec)[3], 0, 0, 0, 0}

#define DEFINE_MMA_SCALE_GFX950_16(AType, BType, EXPAND_A, EXPAND_B, NUM_ACC_A, NUM_ACC_B)                                                                   \
template <typename CompilerTarget>                                                                                                                           \
struct amdgcn_mma<AType, BType, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>> \
: amdgcn_mma_base<AType, BType, fp32_t, 16u, 16u, 128u, 64u, 32, NUM_ACC_A, 1, NUM_ACC_B, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>                               \
{                                                                                                                                                            \
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";                                                      \
    template <typename... Params>                                                                                                                            \
    CK_TILE_DEVICE static CVecType                                                                                                                           \
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t scale_A, int32_t scale_B)                                                 \
    {                                                                                                                                                        \
        using P = WarpGemmParamsParser<Params...>;                                                                                                           \
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(                                                                                            \
            EXPAND_A(aVec),                                                                                                                                  \
            EXPAND_B(bVec),                                                                                                                                  \
            cVec,                                                                                                                                            \
            PackedDataTypeToFlag_v<AType>,                                                                                                                   \
            PackedDataTypeToFlag_v<BType>,                                                                                                                   \
            P::op_sel_a, scale_A,                                                                                                                            \
            P::op_sel_b, scale_B)};                                                                                                                          \
    }                                                                                                                                                        \
};

#define DEFINE_MMA_SCALE_GFX950_32(AType, BType, EXPAND_A, EXPAND_B, NUM_ACC_A, NUM_ACC_B)                                                                  \
template <typename CompilerTarget>                                                                                                                          \
struct amdgcn_mma<AType, BType, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>> \
: amdgcn_mma_base<AType, BType, fp32_t, 32u, 32u, 64u, 64u, 32, NUM_ACC_A, 1, NUM_ACC_B, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>                              \
{                                                                                                                                                           \
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";                                                      \
    template <typename... Params>                                                                                                                           \
    CK_TILE_DEVICE static CVecType                                                                                                                          \
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t scale_A, int32_t scale_B)                                                \
    {                                                                                                                                                       \
        using P = WarpGemmParamsParser<Params...>;                                                                                                          \
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(                                                                                            \
            EXPAND_A(aVec),                                                                                                                                 \
            EXPAND_B(bVec),                                                                                                                                 \
            cVec,                                                                                                                                           \
            PackedDataTypeToFlag_v<AType>,                                                                                                                  \
            PackedDataTypeToFlag_v<BType>,                                                                                                                  \
            P::op_sel_a, scale_A,                                                                                                                           \
            P::op_sel_b, scale_B)};                                                                                                                         \
    }                                                                                                                                                       \
};

// Note on the intrinsic NumAccess values we use here: In principle the "canonical" NumAccess values
// for A and B for gfx950 scale intrinsic is determined by the A and B datatypes. 8-bit datatypes
// require a NumAccess of 2, and 4 and 6-bit types a NumAccess of 1. We follow this *BUT* we do
// allow (1,1) for the cases where A and B are both 8 bit. In these cases, NumAccess (1,1) could
// still be valid when not using scale values.

// 25 intrinsics for __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    fp8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    bf8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    pk_fp6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    pk_bf6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    pk_fp4_t,    MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F4, 1, 1)

// 25 intrinsics for __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    fp8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    bf8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    pk_fp6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    pk_bf6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    pk_fp4_t,    MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F4, 1, 1)

#undef MMA_SCALE_ARG_F8
#undef MMA_SCALE_ARG_F6
#undef MMA_SCALE_ARG_F4
#undef DEFINE_MMA_SCALE_GFX950_16
#undef DEFINE_MMA_SCALE_GFX950_32
// clang-format on

} // namespace ck_tile::core::arch::mma
