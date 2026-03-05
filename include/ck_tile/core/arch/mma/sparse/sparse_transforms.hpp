// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_transforms.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @struct MmaDefaultTransformsSparse
 * @brief Implements the default transforms for Sparse
 *
 * For 2:4 structured sparsity with inline register metadata:
 *  - ATransform: Pass-through (sparse operands formatted in Exec) TODO!
 *  - BTransform: Pass-through (sparse operands already formatted)
 *  - CTransform: Pass-through (input accumulator)
 *  - DTransform: Pass-through (output accumulator as-is)
 */
struct MmaDefaultTransformsSparse
{
    using ATransform = PassThroughTransform;
    using BTransform = PassThroughTransform;
    using CTransform = PassThroughTransform;
    using DTransform = PassThroughTransform;
};

/**
 * @class MmaTransformsDefaultSelector
 * @brief Specialization for Sparse MFMA transforms
 *        Provides default transform selection for sparse operations
 *
 * @tparam MmaOp Sparse MMA operation
 * @tparam CompilerTarget The compiler target
 */
// TODO: c++20 template <MmaOpI MmaOp, amdgcn_target CompilerTarget>
// TODO: c++20 requires(is_mma_op_sparse(MmaOp))
template <typename MmaOp, typename CompilerTarget>
struct MmaTransformsDefaultSelector<MmaOp,
                                    CompilerTarget,
                                    std::enable_if_t<MmaOp::OpFamily == MmaOpFamily::SPARSE>>
{
    using SelectedTransforms = MmaDefaultTransformsSparse;
};

} // namespace ck_tile::core::arch::mma
