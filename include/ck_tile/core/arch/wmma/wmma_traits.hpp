// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "wmma.hpp"

namespace ck_tile::core::arch::wmma {
/** @struct is_mma_op_wmma
 * @brief Trait to check if MmaOp is an WMMA operation
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp, typename = void>
struct is_mma_op_wmma : std::false_type
{
};

/** @struct is_mma_op_wmma
 * @brief MmaOp specialization for WMMA operations, confirming the OpType matches WmmaOp
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp>
struct is_mma_op_wmma<MmaOp, std::enable_if_t<std::is_same_v<typename MmaOp::OpType, WmmaOp>>>
    : std::true_type
{
};

/**
 * @brief Convenience evaluator for is_mma_op_wmma trait
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp>
static constexpr bool is_mma_op_wmma_v = is_mma_op_wmma<MmaOp>::value;

} // namespace ck_tile::core::arch::wmma
