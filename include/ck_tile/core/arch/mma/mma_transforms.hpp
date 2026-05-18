// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/arch/arch.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @struct PassThroughTransform
 * @brief A no-op transform that passes through the input as-is.
 */
struct PassThroughTransform
{
    template <typename VecType>
    CK_TILE_DEVICE static decltype(auto) exec([[clang::lifetimebound]] VecType&& v)
    {
        return std::forward<VecType>(v);
    }
};

/**
 * @struct MmaDefaultPassThroughTransforms
 * @brief Implements the default MMA transforms
 */
struct MmaDefaultPassThroughTransforms
{
    using ATransform = PassThroughTransform;
    using BTransform = PassThroughTransform;
    using CTransform = PassThroughTransform;
    using DTransform = PassThroughTransform;
};

/**
 *  @class MmaTransformsDefaultSelector
 *  @brief  Default selector for MmaTransforms based on MmaOp and CompilerTarget
 *  @tparam MmaOp The Mma operation type
 *  @tparam CompilerTarget The compiler target
 *  @tparam Enable SFINAE parameter for specialization
 */
template <typename MmaOp, typename CompilerTarget, typename Enable = void>
// TODO: c++20 template <MmaOpI MmaOp, amdgcn_target_arch_id CompilerTarget, typename Enable = void>
struct MmaTransformsDefaultSelector
{
    using SelectedTransforms = MmaDefaultPassThroughTransforms;
};

#if CK_TILE_CONCEPTS

/**
 * @concept MmaTransformsI
 * @brief  Expresses the interface of required members for each MmaTransforms type.
 */
template <typename MmaTransforms>
concept MmaTransformsI = requires(MmaTransforms transforms) {
    // Transforms should define TransformA, TransformB, TransformC, and TransformD types
    typename MmaTransforms::ATransform;
    typename MmaTransforms::BTransform;
    typename MmaTransforms::CTransform;
    typename MmaTransforms::DTransform;
};

#endif // CK_TILE_CONCEPTS

} // namespace ck_tile::core::arch::mma
