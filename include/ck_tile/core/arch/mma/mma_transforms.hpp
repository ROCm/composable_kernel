// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

namespace ck::tile::core::arch::mma {

/*! @struct PassThroughTransform
 * @brief A no-op transform that passes through the input as-is.
 */
struct PassThroughTransform
{
    template <typename VecType>
    CK_TILE_DEVICE static decltype(auto) exec(VecType&& v)
    {
        return std::forward<VecType>(v);
    }
};

/*! @struct MmaTransformsDefaultSelector
 *  @brief  Default selector for MmaTransforms based on MmaOp and GfxTargetId.
 *  @tparam MmaOp The Mma operation type.
 *  @tparam GfxTargetId The target architecture ID.
 */
template <MmaOpI MmaOp, uint32_t GfxTargetId, typename Enable = void>
struct MmaTransformsDefaultSelector;

/*! @concept MmaTransformsI
 *  @brief  Expresses the interface of required members for each MmaTransforms type.
 *  @tparam MmaTransforms The MmaTransforms type to be tested.
 */
template <typename MmaTransforms>
concept MmaTransformsI = requires(MmaTransforms transforms) {
    // Transforms should define TransformA, TransformB, TransformC, and TransformD types
    typename MmaTransforms::TransformA;
    typename MmaTransforms::TransformB;
    typename MmaTransforms::TransformC;
    typename MmaTransforms::TransformD;
};

} // namespace ck::tile::core::arch::mma
