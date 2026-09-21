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
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile::core::arch::mma {

namespace scale::detail {

template <typename ValueT, typename T>
inline constexpr int32x16_t to_wmma_scale_arg(const T& vec)
{
    if constexpr(is_any_of<ValueT, fp8_t, bf8_t>::value)
    {
        return bit_cast<int32x16_t>(vec);
    }
    else if constexpr(is_any_of<ValueT, pk_fp6x16_t, pk_bf6x16_t>::value)
    {
        // clang-format off
        return int32x16_t{vec.data[0], vec.data[1], vec.data[2],  vec.data[3],  vec.data[4], vec.data[5], vec.data[6], vec.data[7],
                          vec.data[8], vec.data[9], vec.data[10], vec.data[11], 0, 0, 0, 0};
        // clang-format on
    }
    else if constexpr(is_any_of<ValueT, pk_fp4_t>::value)
    {
        int32x8_t tmp = bit_cast<int32x8_t>(vec);
        return int32x16_t{
            tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], 0, 0, 0, 0, 0, 0, 0, 0};
    }
    else
    {
        static_assert(sizeof(ValueT) == 0, "unsupported ValueT for to_wmma_scale_arg");
    }
}

} // namespace scale::detail

// clang-format off
#define WMMA_SCALE_IMPL(A_TYPE, B_TYPE, NUM_ACC_A, NUM_ACC_B, OP_FAMILY, INSTRUCTION, SCALE_TYPE)                                    \
    template <typename CompilerTarget>                                                                                               \
    struct amdgcn_mma<A_TYPE, B_TYPE, fp32_t, 16u, 16u, 128u, CompilerTarget, OP_FAMILY, enable_if_target_gfx1250_t<CompilerTarget>> \
    : amdgcn_mma_base<A_TYPE, B_TYPE, fp32_t, 16u, 16u, 128u, 32u, 64, NUM_ACC_A, 1, NUM_ACC_B, 1, 8, 1, WmmaOp, OP_FAMILY>          \
    {                                                                                                                                \
        static constexpr const char* instruction_name = #INSTRUCTION;                                                                \
                                                                                                                                     \
        template <typename... Params>                                                                                                \
        CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,                                                                    \
                                            BVecType const& bVec,                                                                    \
                                            CVecType const& cVec,                                                                    \
                                            SCALE_TYPE scaleA,                                                                       \
                                            SCALE_TYPE scaleB)                                                                       \
        {                                                                                                                            \
            using P = WarpGemmParamsParser<Params...>;                                                                               \
            static_assert(                                                                                                           \
                scale::detail::is_legal_combination<A_TYPE, B_TYPE, P::scale_a, P::scale_b>,                                         \
                "Unsupported ADataType/BDataType/scale_a/scale_b combination");                                                      \
            return {INSTRUCTION(PackedDataTypeToFlag_v<A_TYPE>,                                                                      \
                                scale::detail::to_wmma_scale_arg<A_TYPE>(aVec),                                                      \
                                PackedDataTypeToFlag_v<B_TYPE>,                                                                      \
                                scale::detail::to_wmma_scale_arg<B_TYPE>(bVec),                                                      \
                                0,                                                                                                   \
                                cVec,                                                                                                \
                                P::op_sel_a,                                                                                         \
                                P::scale_a,                                                                                          \
                                scaleA,                                                                                              \
                                P::op_sel_b,                                                                                         \
                                P::scale_b,                                                                                          \
                                scaleB,                                                                                              \
                                P::reuse_a,                                                                                          \
                                P::reuse_b)};                                                                                        \
        }                                                                                                                            \
    };
#define WMMA_SCALE32_IMPL(A_TYPE, B_TYPE, NUM_ACC_A, NUM_ACC_B)       \
    WMMA_SCALE_IMPL(A_TYPE,                                           \
                    B_TYPE,                                           \
                    NUM_ACC_A,                                        \
                    NUM_ACC_B,                                        \
                    MmaOpFamily::SCALE,                               \
                    __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, \
                    int32_t)
#define WMMA_SCALE16_IMPL(A_TYPE, B_TYPE, NUM_ACC_A, NUM_ACC_B)         \
    WMMA_SCALE_IMPL(A_TYPE,                                             \
                    B_TYPE,                                             \
                    NUM_ACC_A,                                          \
                    NUM_ACC_B,                                          \
                    MmaOpFamily::SCALE16,                               \
                    __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, \
                    int64_t)

WMMA_SCALE32_IMPL(fp8_t,       fp8_t,       1, 1)
WMMA_SCALE32_IMPL(fp8_t,       bf8_t,       1, 1)
WMMA_SCALE32_IMPL(fp8_t,       pk_fp6x16_t, 4, 2)
WMMA_SCALE32_IMPL(fp8_t,       pk_bf6x16_t, 4, 2)
WMMA_SCALE32_IMPL(fp8_t,       pk_fp4_t,    4, 2)
WMMA_SCALE32_IMPL(bf8_t,       fp8_t,       1, 1)
WMMA_SCALE32_IMPL(bf8_t,       bf8_t,       1, 1)
WMMA_SCALE32_IMPL(bf8_t,       pk_fp6x16_t, 4, 2)
WMMA_SCALE32_IMPL(bf8_t,       pk_bf6x16_t, 4, 2)
WMMA_SCALE32_IMPL(bf8_t,       pk_fp4_t,    4, 2)
WMMA_SCALE32_IMPL(pk_fp6x16_t, fp8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_fp6x16_t, bf8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_fp6x16_t, pk_fp6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_fp6x16_t, pk_bf6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_fp6x16_t, pk_fp4_t,    1, 1)
WMMA_SCALE32_IMPL(pk_bf6x16_t, fp8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_bf6x16_t, bf8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_bf6x16_t, pk_fp6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_bf6x16_t, pk_bf6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_bf6x16_t, pk_fp4_t,    1, 1)
WMMA_SCALE32_IMPL(pk_fp4_t,    fp8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_fp4_t,    bf8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_fp4_t,    pk_fp6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_fp4_t,    pk_bf6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_fp4_t,    pk_fp4_t,    1, 1)

#undef WMMA_SCALE32_IMPL

WMMA_SCALE16_IMPL(fp8_t,       fp8_t,       1, 1)
WMMA_SCALE16_IMPL(fp8_t,       bf8_t,       1, 1)
WMMA_SCALE16_IMPL(fp8_t,       pk_fp6x16_t, 4, 2)
WMMA_SCALE16_IMPL(fp8_t,       pk_bf6x16_t, 4, 2)
WMMA_SCALE16_IMPL(fp8_t,       pk_fp4_t,    4, 2)
WMMA_SCALE16_IMPL(bf8_t,       fp8_t,       1, 1)
WMMA_SCALE16_IMPL(bf8_t,       bf8_t,       1, 1)
WMMA_SCALE16_IMPL(bf8_t,       pk_fp6x16_t, 4, 2)
WMMA_SCALE16_IMPL(bf8_t,       pk_bf6x16_t, 4, 2)
WMMA_SCALE16_IMPL(bf8_t,       pk_fp4_t,    4, 2)
WMMA_SCALE16_IMPL(pk_fp6x16_t, fp8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_fp6x16_t, bf8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_fp6x16_t, pk_fp6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_fp6x16_t, pk_bf6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_fp6x16_t, pk_fp4_t,    1, 1)
WMMA_SCALE16_IMPL(pk_bf6x16_t, fp8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_bf6x16_t, bf8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_bf6x16_t, pk_fp6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_bf6x16_t, pk_bf6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_bf6x16_t, pk_fp4_t,    1, 1)
WMMA_SCALE16_IMPL(pk_fp4_t,    fp8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_fp4_t,    bf8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_fp4_t,    pk_fp6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_fp4_t,    pk_bf6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_fp4_t,    pk_fp4_t,    1, 1)

#undef WMMA_SCALE16_IMPL
#undef WMMA_SCALE_IMPL
// clang-format on

} // namespace ck_tile::core::arch::mma
