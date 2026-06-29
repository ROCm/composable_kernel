// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @defgroup sparse_mfma_gfx9 Sparse MFMA for GFX9
 * @brief Sparse specializations of @ref amdgcn_mma for GFX9 family.
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

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x32_f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_smfmac_f32_16x16x32_f16(
            aVec,
            bVec,
            cVec,
            idx,
            P::cbsz,  // Ignore abid and use first portion Y/N
            P::abid); // Portion of idx VGPR containing idx info
    };
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x16_f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_f32_32x32x16_f16(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x32_bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_f32_16x16x32_bf16(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x16_bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_f32_32x32x16_bf16(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                      |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_i32_16x16x64_i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_i32_16x16x64_i8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                      |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_i32_32x32x32_i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_i32_32x32x32_i8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 32u, 32u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 32u, 32u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_f32_16x16x64_f16(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 32u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_f32_32x32x32_f16(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x64_bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_f32_16x16x64_bf16(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x32_bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_f32_32x32x32_bf16(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK            |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                       |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_i32_16x16x128_i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_i32_16x16x128_i8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                      |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_i32_32x32x64_i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_smfmac_i32_32x32x64_i8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8(aVec, bVec, cVec, idx, P::cbsz, P::abid)};
    }
};

/** @} */ // sparse_mfma_gfx9

} // namespace ck_tile::core::arch::mma
