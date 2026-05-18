// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"

#include <cstdint>
#include <stdio.h>
#include <type_traits>
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>
#endif

namespace ck_tile::core::arch::mma {

/**
 * @struct MfmaOp
 * @brief Meta-tag for the MFMA operation. This will be used in the MmaOp policies to
 * identify the operation as an MFMA operation.
 */
struct MfmaOp
{
};

CK_TILE_HOST_DEVICE constexpr const char* to_string(MfmaOp) { return "MfmaOp"; }

/**
 * @class is_mma_op_mfma
 * @brief Trait to check if MmaOp is an MFMA operation
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp, typename = void>
struct is_mma_op_mfma : std::false_type
{
};

/**
 * @struct is_mma_op_mfma
 * @brief MmaOp specialization for MFMA operations, confirming the OpType matches MfmaOp
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp>
// TODO: c++20 requires
struct is_mma_op_mfma<MmaOp, std::enable_if_t<std::is_same_v<typename MmaOp::OpType, MfmaOp>>>
    : std::true_type
{
};

/**
 * @brief Convenience evaluator for is_mma_op_mfma trait
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp>
static constexpr bool is_mma_op_mfma_v = is_mma_op_mfma<MmaOp>::value;

/**
 * @struct DefaultMfmaCtrlFlags
 * @brief Default MFMA flags, no broadcasting or rotation of inputs
 */
struct DefaultMfmaCtrlFlags
{
    static constexpr uint32_t Cbsz = 0; // CBSZ flag, default 0
    static constexpr uint32_t Abid = 0; // ABID flag, default 0
    static constexpr uint32_t Blgp = 0; // BLGP flag, default 0
};

CK_TILE_HOST_DEVICE void print_flags(DefaultMfmaCtrlFlags const& ctrlFlags)
{
    printf("CtrlFlags      Cbsz / Abid / Blgp       : %u / %u / %u\n",
           ctrlFlags.Cbsz,
           ctrlFlags.Abid,
           ctrlFlags.Blgp);
}

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept CtrlFlagsGfx9I
 * @brief  Expresses the interface of required members for each CtrlFlags type on Gfx9
 */
template <typename CtrlFlags>
concept CtrlFlagsGfx9I = requires(CtrlFlags ctrlFlags) {
    // Flag members for Gfx9 MFMA instructions
    { CtrlFlags::Cbsz } -> std::convertible_to<int>;
    { CtrlFlags::Abid } -> std::convertible_to<int>;
    { CtrlFlags::Blgp } -> std::convertible_to<int>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
