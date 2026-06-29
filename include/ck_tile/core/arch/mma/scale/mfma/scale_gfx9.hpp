// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/scale/scale_traits.hpp"
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

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 2, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        int32_t scale_A,
                                        int32_t scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 2, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        int32_t scale_A,
                                        int32_t scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes                 |MNK            |
struct amdgcn_mma<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                                |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        int32_t scale_A,
                                        int32_t scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            int32x8_t{aVec.data[0], aVec.data[1], aVec.data[2], aVec.data[3], aVec.data[4], aVec.data[5], 0, 0},
            int32x8_t{bVec.data[0], bVec.data[1], bVec.data[2], bVec.data[3], bVec.data[4], bVec.data[5], 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp6x16_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp6x16_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};
// clang-format on

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes                 |MNK            |
struct amdgcn_mma<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                                |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        int32_t scale_A,
                                        int32_t scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            int32x8_t{aVec.data[0], aVec.data[1], aVec.data[2], aVec.data[3], aVec.data[4], aVec.data[5], 0, 0},
            int32x8_t{bVec.data[0], bVec.data[1], bVec.data[2], bVec.data[3], bVec.data[4], bVec.data[5], 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_bf6x16_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_bf6x16_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};
// clang-format on

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes           |MNK            |
struct amdgcn_mma<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                          |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        int32_t scale_A,
                                        int32_t scale_B)
    {
        using P         = WarpGemmParamsParser<Params...>;
        int32x4_t arg_a = bit_cast<int32x4_t>(aVec);
        int32x4_t arg_b = bit_cast<int32x4_t>(bVec);

        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            int32x8_t{arg_a[0], arg_a[1], arg_a[2], arg_a[3], 0, 0, 0, 0},
            int32x8_t{arg_b[0], arg_b[1], arg_b[2], arg_b[3], 0, 0, 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp4_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp4_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 2, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        int32_t scale_A,
                                        int32_t scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                   |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 2, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        int32_t scale_A,
                                        int32_t scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes                 |MNK           |
struct amdgcn_mma<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                               |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        int32_t scale_A,
                                        int32_t scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            int32x8_t{aVec.data[0], aVec.data[1], aVec.data[2], aVec.data[3], aVec.data[4], aVec.data[5], 0, 0},
            int32x8_t{bVec.data[0], bVec.data[1], bVec.data[2], bVec.data[3], bVec.data[4], bVec.data[5], 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp6x16_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp6x16_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};
// clang-format on

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes                 |MNK           |
struct amdgcn_mma<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                               |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        int32_t scale_A,
                                        int32_t scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            int32x8_t{aVec.data[0], aVec.data[1], aVec.data[2], aVec.data[3], aVec.data[4], aVec.data[5], 0, 0},
            int32x8_t{bVec.data[0], bVec.data[1], bVec.data[2], bVec.data[3], bVec.data[4], bVec.data[5], 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_bf6x16_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_bf6x16_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};
// clang-format on

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes           |MNK           |
struct amdgcn_mma<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
//                                                         |WS  |AParams  |BPar |CPar  |
: amdgcn_mma_base<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,
                                        BVecType const& bVec,
                                        CVecType const& cVec,
                                        int32_t scale_A,
                                        int32_t scale_B)
    {
        using P         = WarpGemmParamsParser<Params...>;
        int32x4_t arg_a = bit_cast<int32x4_t>(aVec);
        int32x4_t arg_b = bit_cast<int32x4_t>(bVec);

        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            int32x8_t{arg_a[0], arg_a[1], arg_a[2], arg_a[3], 0, 0, 0, 0},
            int32x8_t{arg_b[0], arg_b[1], arg_b[2], arg_b[3], 0, 0, 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp4_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp4_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

/** @} */ // scale_mfma_gfx9

} // namespace ck_tile::core::arch::mma
