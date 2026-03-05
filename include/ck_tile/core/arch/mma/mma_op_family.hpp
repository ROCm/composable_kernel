// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

namespace ck_tile::core::arch::mma {

/**
 * @enum MmaOpFamily
 * @brief Enumeration that defines mma op families and
 */
enum struct MmaOpFamily
{
    UNDEFINED = 0,
    DENSE,
    SPARSE,
    SCALE,
};

/**
 * @class is_ctrl_fis_mma_op_of_familylag_of_family
 * @brief Meta-function to check if MmaOp is of the specified MmaOpFamily
 * @tparam Family Control flag family
 * @tparam MmaOp amdgcn struct specialization type
 */
template <MmaOpFamily Family, typename MmaOp, typename = void>
struct is_mma_op_of_family : std::false_type
{
};

/**
 * @struct is_mma_op_of_family
 * @brief Specialization for Family == MmaOp::OpFamily detection
 */
template <MmaOpFamily Family, typename MmaOp>
struct is_mma_op_of_family<Family, MmaOp, std::enable_if_t<Family == MmaOp::OpFamily>>
    : std::true_type
{
};

/**
 * @brief Convenience evaluator for is_mma_op_of_family trait
 * @tparam Family Desired control flag family
 * @tparam MmaOp The amdgcn struct specialization type to check
 */
template <MmaOpFamily Family, typename MmaOp>
static constexpr bool is_mma_op_of_family_v = is_mma_op_of_family<Family, MmaOp>::value;

} // namespace ck_tile::core::arch::mma
