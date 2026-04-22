// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_transforms.hpp"

#include <type_traits>

namespace ck_tile::core::arch::mma {

/**
 * @struct MmaDefaultTransformsScale
 * @brief Implements the default MMA transforms for Scale
 */
struct MmaDefaultTransformsScale
{
    using ATransform = PassThroughTransform;
    using BTransform = PassThroughTransform;
    using CTransform = PassThroughTransform;
    using DTransform = PassThroughTransform;
};

/**
 * @struct MmaTransformsDefaultSelector
 * @brief Specialization for Scale MFMA transforms
 *        Provides default transform selection for scale operations
 *
 * @tparam MmaOp Scale MMA operation
 * @tparam CompilerTarget The compiler target
 */
// TODO: c++20 template <MmaOpI MmaOp, amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires(is_mma_op_scale(MmaOp))
template <typename MmaOp, typename CompilerTarget>
struct MmaTransformsDefaultSelector<MmaOp,
                                    CompilerTarget,
                                    std::enable_if_t<MmaOp::OpFamily == MmaOpFamily::SCALE>>
{
    using SelectedTransforms = MmaDefaultTransformsScale;
};

} // namespace ck_tile::core::arch::mma
