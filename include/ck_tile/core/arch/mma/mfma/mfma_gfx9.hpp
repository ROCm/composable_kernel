// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "mfma_traits.hpp"

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/tfloat32.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/ignore.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile::core::arch::mma {

// NOTE: At this point forward, we are specializing amdgcn_mma for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

/**
 * @defgroup dense_mfma_gfx9 Dense MFMA for GFX9
 * @brief Dense specializations of @ref amdgcn_mma for GFX9 family.
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
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 32u, 64u, 1u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 32u, 64u, 1u, 64u, 1, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x1f32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x1f32(
            bit_cast<fp32_t>(aVec), bit_cast<fp32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 64u, 32u, 1u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 64u, 32u, 1u, 64u, 1, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x1f32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x1f32(
            bit_cast<fp32_t>(aVec), bit_cast<fp32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 16u, 64u, 1u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 16u, 64u, 1u, 64u, 1, 1, 4, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x1f32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x1f32(
            bit_cast<fp32_t>(aVec), bit_cast<fp32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 64u, 16u, 1u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 64u, 16u, 1u, 64u, 1, 1, 1, 1, 4, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x1f32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x1f32(
            bit_cast<fp32_t>(aVec), bit_cast<fp32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK         |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 4u, 64u, 1u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 4u, 64u, 1u, 64u, 1, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x1f32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_4x4x1f32(
            bit_cast<fp32_t>(aVec), bit_cast<fp32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK         |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 64u, 4u, 1u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                   |WS  |AParams |BPar  |CPar |
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 64u, 4u, 1u, 64u, 1, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x1f32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_4x4x1f32(
            bit_cast<fp32_t>(aVec), bit_cast<fp32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 32u, 32u, 2u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 32u, 32u, 2u, 64u, 1, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x2f32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x2f32(
            bit_cast<fp32_t>(aVec), bit_cast<fp32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 16u, 16u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 16u, 16u, 4u, 64u, 1, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x4f32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x4f32(
            bit_cast<fp32_t>(aVec), bit_cast<fp32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 64u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 64u, 4u, 64u, 4, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x4f16(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 32u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 32u, 4u, 64u, 4, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x4f16(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 64u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 64u, 4u, 64u, 4, 1, 4, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x4f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x4f16(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 16u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 16u, 4u, 64u, 4, 1, 1, 1, 4, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x4f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x4f16(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK         |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 4u, 64u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 4u, 64u, 4u, 64u, 4, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x4f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_4x4x4f16(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK         |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 4u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                   |WS  |AParams |BPar  |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 4u, 4u, 64u, 4, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x4f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_4x4x4f16(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 32u, 8u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 32u, 8u, 64u, 4, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x8f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x8f16(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                     |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, 64u, 4, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x16f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x16f16(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK          |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 64u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                     |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 64u, 4u, 64u, 4, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_32x32x4i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_32x32x4i8(
            bit_cast<int32_t>(aVec), bit_cast<int32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK          |
struct amdgcn_mma<int8_t, int8_t, int32_t, 64u, 32u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                     |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 64u, 32u, 4u, 64u, 4, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_32x32x4i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_32x32x4i8(
            bit_cast<int32_t>(aVec), bit_cast<int32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK          |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 64u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                     |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 64u, 4u, 64u, 4, 1, 4, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_16x16x4i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_16x16x4i8(
            bit_cast<int32_t>(aVec), bit_cast<int32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK          |
struct amdgcn_mma<int8_t, int8_t, int32_t, 64u, 16u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                     |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 64u, 16u, 4u, 64u, 4, 1, 1, 1, 4, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_16x16x4i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_16x16x4i8(
            bit_cast<int32_t>(aVec), bit_cast<int32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK         |
struct amdgcn_mma<int8_t, int8_t, int32_t, 4u, 64u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 4u, 64u, 4u, 64u, 4, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_4x4x4i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_4x4x4i8(
            bit_cast<int32_t>(aVec), bit_cast<int32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK         |
struct amdgcn_mma<int8_t, int8_t, int32_t, 64u, 4u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar  |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 64u, 4u, 4u, 64u, 4, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_4x4x4i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_4x4x4i8(
            bit_cast<int32_t>(aVec), bit_cast<int32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK          |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 8u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                     |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 8u, 64u, 4, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_32x32x8i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_32x32x8i8(
            bit_cast<int32_t>(aVec), bit_cast<int32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                      |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 16u, 64u, 4, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_16x16x16i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_16x16x16i8(
            bit_cast<int32_t>(aVec), bit_cast<int32_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

// --------------------------------------------------------------------------------------------
// TEMPORARY gfx908/gfx90a int8 ASM-PARITY SHIMS -- throwaway, revisit and remove.
//
// gfx908/gfx90a only have native i8 MFMA at 32x32x8 / 16x16x16 (the two specializations above),
// which accumulate natively in i32. The LEGACY warp-gemm path instead dispatches i8 through the
// gfx942-shaped 32x32x16 / 16x16x32 warp gemms, whose gfx90a fallbacks are a DIFFERENT compute:
//   * 32x32x16 -> upcast i8->f32 and issue 8x v_mfma_f32_32x32x2f32 (f32 accumulation), and
//   * 16x16x32 -> no native intrinsic, i.e. a no-op that produces zeros.
// (See WarpGemmAttributeMfmaImpl_i32_32x32x16_i8 / _i32_16x16x32_i8 in
//  ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp.)
//
// These two shims reproduce the legacy shapes+compute so the new framework's gfx90a i8 GPU asm
// matches the legacy (USE_NEW_UNIFIED_FRAMEWORK=0) build. Because MmaKSearchSelector picks the
// largest intrinsic K that fits WaveTileK, adding 32x32x16 / 16x16x32 makes the selector prefer
// them over the native 32x32x8 / 16x16x16, reproducing the legacy codegen.
//
// WARNING: this is a DELIBERATE DEGRADATION for asm-parity validation only. It changes gfx90a i8
// numerics (native i32 accumulation -> f32 upcast) and reintroduces the 16x16x32 no-op. It is
// confined to the USE_NEW_UNIFIED_FRAMEWORK=1 path and does not touch the legacy build. REMOVE
// once the native-i8 divergence has been triaged (see TODO in the change summary).
// --------------------------------------------------------------------------------------------
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                      |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "i8 32x32x16 gfx90a legacy-parity shim (f32 upcast, 8x v_mfma_f32_32x32x2f32)";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        // Mirror WarpGemmAttributeMfmaImpl_i32_32x32x16_i8's gfx908/gfx90a fallback exactly so the
        // emitted f32 MFMA sequence matches the legacy build.
        CVecType c = cVec;
        static_for<0, 8, 1>{}([&](auto k) {
            float a_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<ADataType, 8>&>(aVec)
                                        .template get_as<ADataType>()[number<k>{}]);
            float b_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<BDataType, 8>&>(bVec)
                                        .template get_as<BDataType>()[number<k>{}]);
            c = __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, c, 0, 0, 0);
        });
        return c;
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                      |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "i8 16x16x32 gfx90a legacy-parity shim (no-op)";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        // Mirror WarpGemmAttributeMfmaImpl_i32_16x16x32_i8: no native gfx90a intrinsic, so the
        // legacy build emits nothing for the multiply -- reproduce that no-op here.
        ck_tile::ignore = aVec;
        ck_tile::ignore = bVec;
        return cVec;
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 64u, 2u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 64u, 2u, 64u, 2, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x2bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x2bf16(
            bit_cast<int16x2_t>(aVec), bit_cast<int16x2_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 32u, 2u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 32u, 2u, 64u, 2, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x2bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x2bf16(
            bit_cast<int16x2_t>(aVec), bit_cast<int16x2_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 64u, 2u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 64u, 2u, 64u, 2, 1, 4, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x2bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x2bf16(
            bit_cast<int16x2_t>(aVec), bit_cast<int16x2_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 16u, 2u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 16u, 2u, 64u, 2, 1, 1, 1, 4, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x2bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x2bf16(
            bit_cast<int16x2_t>(aVec), bit_cast<int16x2_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK         |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 4u, 64u, 2u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 4u, 64u, 2u, 64u, 2, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x2bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_4x4x2bf16(
            bit_cast<int16x2_t>(aVec), bit_cast<int16x2_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK         |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 4u, 2u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                   |WS  |AParams |BPar  |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 4u, 2u, 64u, 2, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x2bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_4x4x2bf16(
            bit_cast<int16x2_t>(aVec), bit_cast<int16x2_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 4u, 64u, 2, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x4bf16(
            bit_cast<int16x2_t>(aVec), bit_cast<int16x2_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 8u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 8u, 64u, 2, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x8bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x8bf16(
            bit_cast<int16x2_t>(aVec), bit_cast<int16x2_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 64u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 64u, 4u, 64u, 4, 1, 2, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4bf16_1k";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x4bf16_1k(
            bit_cast<int16x4_t>(aVec), bit_cast<int16x4_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 32u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 32u, 4u, 64u, 4, 1, 1, 1, 2, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4bf16_1k";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x4bf16_1k(
            bit_cast<int16x4_t>(aVec), bit_cast<int16x4_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 64u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 64u, 4u, 64u, 4, 1, 4, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x4bf16_1k";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x4bf16_1k(
            bit_cast<int16x4_t>(aVec), bit_cast<int16x4_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 16u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 16u, 4u, 64u, 4, 1, 1, 1, 4, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x4bf16_1k";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x4bf16_1k(
            bit_cast<int16x4_t>(aVec), bit_cast<int16x4_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK         |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 4u, 64u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 4u, 64u, 4u, 64u, 4, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x4bf16_1k";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
            bit_cast<int16x4_t>(aVec), bit_cast<int16x4_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK         |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 4u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar  |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 4u, 4u, 64u, 4, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_4x4x4bf16_1k";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
            bit_cast<int16x4_t>(aVec), bit_cast<int16x4_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 8u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 8u, 64u, 4, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x8bf16_1k";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
            bit_cast<int16x4_t>(aVec), bit_cast<int16x4_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, 64u, 4, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x16bf16_1k";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
            bit_cast<int16x4_t>(aVec), bit_cast<int16x4_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp64_t, fp64_t, fp64_t, 16u, 16u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp64_t, fp64_t, fp64_t, 16u, 16u, 4u, 64u, 1, 1, 1, 1, 1, 4, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f64_16x16x4f64";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        // Note: BLGP flag has another meaning for f64 builtins: BLGP bits [0:2] cause negation of
        // the A, B, and C input matrices respectively (ref. ISA docs for MI300 Instinct)
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f64_16x16x4f64(bit_cast<fp64_t>(aVec),
                                                     bit_cast<fp64_t>(bVec),
                                                     cVec,
                                                     P::cbsz, // CBSZ ignored for f64
                                                     P::abid, // ABID ignored for f64
                                                     P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK         |
struct amdgcn_mma<fp64_t, fp64_t, fp64_t, 4u, 16u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp64_t, fp64_t, fp64_t, 4u, 16u, 4u, 64u, 1, 1, 4, 1, 1, 1, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f64_4x4x4f64";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f64_4x4x4f64(bit_cast<fp64_t>(aVec),
                                                   bit_cast<fp64_t>(bVec),
                                                   bit_cast<fp64_t>(cVec),
                                                   P::cbsz, // CBSZ ignored for f64
                                                   P::abid, // ABID ignored for f64
                                                   P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK         |
struct amdgcn_mma<fp64_t, fp64_t, fp64_t, 16u, 4u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX90A, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp64_t, fp64_t, fp64_t, 16u, 4u, 4u, 64u, 1, 1, 1, 1, 4, 1, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f64_4x4x4f64";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f64_4x4x4f64(bit_cast<fp64_t>(aVec),
                                                   bit_cast<fp64_t>(bVec),
                                                   bit_cast<fp64_t>(cVec),
                                                   P::cbsz, // CBSZ ignored for f64
                                                   P::abid, // ABID ignored for f64
                                                   P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                      |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_16x16x32_i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_16x16x32_i8(
            bit_cast<int64_t>(aVec), bit_cast<int64_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                      |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_32x32x16_i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_32x32x16_i8(
            bit_cast<int64_t>(aVec), bit_cast<int64_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<tf32_t, tf32_t, fp32_t, 16u, 16u, 8u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<tf32_t, tf32_t, fp32_t, 16u, 16u, 8u, 64u, 2, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x8_xf32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_mfma_f32_16x16x8_xf32(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<tf32_t, tf32_t, fp32_t, 32u, 32u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942>>
//                                                    |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<tf32_t, tf32_t, fp32_t, 32u, 32u, 4u, 64u, 2, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x4_xf32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_mfma_f32_32x32x4_xf32(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(
            bit_cast<int64_t>(aVec), bit_cast<int64_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(
            bit_cast<int64_t>(aVec), bit_cast<int64_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(
            bit_cast<int64_t>(aVec), bit_cast<int64_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            bit_cast<int64_t>(aVec), bit_cast<int64_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
            bit_cast<int64_t>(aVec), bit_cast<int64_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
            bit_cast<int64_t>(aVec), bit_cast<int64_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
            bit_cast<int64_t>(aVec), bit_cast<int64_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
            bit_cast<int64_t>(aVec), bit_cast<int64_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_mfma_f32_16x16x32_f16(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_16x16x32_bf16(bit_cast<llvm_bf16x8_t>(aVec),
                                                        bit_cast<llvm_bf16x8_t>(bVec),
                                                        cVec,
                                                        P::cbsz,
                                                        P::abid,
                                                        P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {
            __builtin_amdgcn_mfma_f32_32x32x16_f16(aVec, bVec, cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_f32_32x32x16_bf16(bit_cast<llvm_bf16x8_t>(aVec),
                                                        bit_cast<llvm_bf16x8_t>(bVec),
                                                        cVec,
                                                        P::cbsz,
                                                        P::abid,
                                                        P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                      |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 64u, 64u, 16, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_16x16x64_i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_16x16x64_i8(
            bit_cast<int32x4_t>(aVec), bit_cast<int32x4_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                      |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 32u, 64u, 16, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_i32_32x32x32_i8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_i32_32x32x32_i8(
            bit_cast<int32x4_t>(aVec), bit_cast<int32x4_t>(bVec), cVec, P::cbsz, P::abid, P::blgp)};
    }
};

// ------------------------------------------------------------------------------------------------
// -------------------------------- WORKAROUND INTRINSICS -----------------------------------------
// ------------------------------------------------------------------------------------------------
// The following are intrinsic wrappers that do not directly represent a single intrinsic, but
// rather a modified intrinsic call, multiple intrinsic calls, or a dummy wrapper. A modified
// intrinsic call may combine an intrinsic call with a conversion operation for example. Multiple
// intrinsic calls may be used to construct a larger effective intrinsic from multiple smaller ones
// (and maybe also perform conversions). Dummy amdgcn_structs may either not have an exec function
// at all and be used only for layout parameters, or have a dummy exec function that does nothing.
// Some rare CK Tile tests depend on these.

// Custom version of fp8xfp8 16x16x32 using dummy exec for gfx908 and gfx90a.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8_gfx908_gfx90a_not_supported";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        ck_tile::ignore = aVec;
        ck_tile::ignore = bVec;
        return cVec;
    }
};

// Custom version of fp8xfp8 32x32x16 using multi-intrinsic workaround for gfx908 and gfx90a.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                   |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8_gfx908_gfx90a_multi_intrinsic_workaround";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec, BVecType const& bVec, CVecType& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        static_for<0, 8, 1>{}([&](auto k) {
            float a_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<ADataType, 8>&>(aVec)
                                        .template get_as<ADataType>()[number<k>{}]);
            float b_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<BDataType, 8>&>(bVec)
                                        .template get_as<BDataType>()[number<k>{}]);

            cVec =
                __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, cVec, P::cbsz, P::abid, P::blgp);
        });
        return cVec;
    }
};

// Layout-only placeholder for an unsupported fp8xfp8 16x16x16 operation on gfx9.
// Its WMMA-like layout parameters keep legacy layout/dispatch queries well-formed;
// the missing exec() prevents it from being used as an instruction.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx9_not_supported";
};

// Custom version of fp8xbf8 16x16x32 using dummy exec for gfx908 and gfx90a.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8_gfx908_gfx90a_not_supported";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        ck_tile::ignore = aVec;
        ck_tile::ignore = bVec;
        return cVec;
    }
};

// Custom version of fp8xbf8 32x32x16 using multi-intrinsic workaround for gfx908 and gfx90a.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                   |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8_gfx908_gfx90a_multi_intrinsic_workaround";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec, BVecType const& bVec, CVecType& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        static_for<0, 8, 1>{}([&](auto k) {
            float a_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<ADataType, 8>&>(aVec)
                                        .template get_as<ADataType>()[number<k>{}]);
            float b_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<BDataType, 8>&>(bVec)
                                        .template get_as<BDataType>()[number<k>{}]);

            cVec =
                __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, cVec, P::cbsz, P::abid, P::blgp);
        });
        return cVec;
    }
};

// Layout-only placeholder for an unsupported fp8xbf8 16x16x16 operation on gfx9.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx9_not_supported";
};

// Custom version of bf8xfp8 16x16x32 using dummy exec for gfx908 and gfx90a.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8_gfx908_gfx90a_not_supported";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        ck_tile::ignore = aVec;
        ck_tile::ignore = bVec;
        return cVec;
    }
};

// Custom version of bf8xfp8 32x32x16 using multi-intrinsic workaround for gfx908 and gfx90a.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                   |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8_gfx908_gfx90a_multi_intrinsic_workaround";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec, BVecType const& bVec, CVecType& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        static_for<0, 8, 1>{}([&](auto k) {
            float a_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<ADataType, 8>&>(aVec)
                                        .template get_as<ADataType>()[number<k>{}]);
            float b_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<BDataType, 8>&>(bVec)
                                        .template get_as<BDataType>()[number<k>{}]);

            cVec =
                __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, cVec, P::cbsz, P::abid, P::blgp);
        });
        return cVec;
    }
};

// Layout-only placeholder for an unsupported bf8xfp8 16x16x16 operation on gfx9.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx9_not_supported";
};

// Custom version of bf8xbf8 16x16x32 using dummy exec for gfx908 and gfx90a.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8_gfx908_gfx90a_not_supported";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        ck_tile::ignore = aVec;
        ck_tile::ignore = bVec;
        return cVec;
    }
};

// Custom version of bf8xbf8 32x32x16 using multi-intrinsic workaround for gfx908 and gfx90a.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX908, amdgcn_target_id::GFX90A>>
//                                                   |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8_gfx908_gfx90a_multi_intrinsic_workaround";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec, BVecType const& bVec, CVecType& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        static_for<0, 8, 1>{}([&](auto k) {
            float a_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<ADataType, 8>&>(aVec)
                                        .template get_as<ADataType>()[number<k>{}]);
            float b_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<BDataType, 8>&>(bVec)
                                        .template get_as<BDataType>()[number<k>{}]);

            cVec =
                __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, cVec, P::cbsz, P::abid, P::blgp);
        });
        return cVec;
    }
};

// Layout-only placeholder for an unsupported bf8xbf8 16x16x16 operation on gfx9.
template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
//                                                   |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx9_not_supported";
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           | 
struct amdgcn_mma<tf32_t, tf32_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<tf32_t, tf32_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_16x16x32_bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        ext_vector_t<bf16_t, 8> a_big;
        ext_vector_t<bf16_t, 8> a_small;
        ext_vector_t<bf16_t, 8> b_big;
        ext_vector_t<bf16_t, 8> b_small;
        convert_float_to_bf16_pairs<8>(aVec, a_big, a_small);
        convert_float_to_bf16_pairs<8>(bVec, b_big, b_small);

        using P = WarpGemmParamsParser<Params...>;

        auto result = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            a_small, b_big, cVec, P::cbsz, P::abid, P::blgp);
        result = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            a_big, b_small, result, P::cbsz, P::abid, P::blgp);
        result = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            a_big, b_big, result, P::cbsz, P::abid, P::blgp);
        return {result};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<tf32_t, tf32_t, fp32_t, 32u, 32u, 16u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                     |WS  |AParams |BPar |CPar  |
: amdgcn_mma_base<tf32_t, tf32_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_f32_32x32x16_bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        ext_vector_t<bf16_t, 8> a_big;
        ext_vector_t<bf16_t, 8> a_small;
        ext_vector_t<bf16_t, 8> b_big;
        ext_vector_t<bf16_t, 8> b_small;
        convert_float_to_bf16_pairs<8>(aVec, a_big, a_small);
        convert_float_to_bf16_pairs<8>(bVec, b_big, b_small);

        using P = WarpGemmParamsParser<Params...>;

        auto result = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
            a_small, b_big, cVec, P::cbsz, P::abid, P::blgp);
        result = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
            a_big, b_small, result, P::cbsz, P::abid, P::blgp);
        result = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
            a_big, b_big, result, P::cbsz, P::abid, P::blgp);
        return {result};
    }
};

/** @} */ // dense_mfma_gfx9

} // namespace ck_tile::core::arch::mma
