// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "wmma.hpp"
namespace ck::tile::core::arch::mma {

/*! @struct DuplicateTransform
 * @brief Transform to duplicate low register elements to high register elements
 */
struct DuplicateTransform
{
    template <typename VecType>
    CK_TILE_DEVICE static decltype(auto) exec(VecType&& v)
    {
        // TODO: Implement duplication logic to broadcast low
        // register elements to high elements [0 - (N/2 -1)] -> [N/2 - (N-1)]
        return std::forward<VecType>(v);
    }
};

/*! @struct PadTransform
 * @brief Transform to pad data from original type to b32 type
 */
struct PadTransform
{
    template <typename VecType>
    CK_TILE_DEVICE static decltype(auto) exec(VecType&& v)
    {
        // TODO: Implement b32 padding logic.
        // E.g., for fp16, pad each 16-bit element with 16 bits of 0 to make 32-bit elements
        return std::forward<VecType>(v);
    }
};

/*! @struct UnpadTransform
 * @brief Transform to unpad data from b32 type to original type
 */
struct UnpadTransform
{
    template <typename VecType>
    CK_TILE_DEVICE static decltype(auto) exec(VecType&& v)
    {
        // TODO: Implement b32 logic to unpad 32 to original data type.
        return std::forward<VecType>(v);
    }
};

/*! @struct MmaDefaultTransformsGfx11
 * @brief Default MMA transforms for GFX11 architecture
 */
struct MmaDefaultTransformsGfx11
{
    using TransformA = DuplicateTransform;
    using TransformB = DuplicateTransform;
    using TransformC = PadTransform;
    using TransformD = UnpadTransform;
};

/*! @struct MmaDefaultTransformsGfx12
 * @brief Default MMA transforms for GFX12 architecture
 */
struct MmaDefaultTransformsGfx12
{
    using TransformA = PassThroughTransform;
    using TransformB = PassThroughTransform;
    using TransformC = PassThroughTransform;
    using TransformD = PassThroughTransform;
};

/*! @struct MmaTransformsDefaultSelector
 * @brief Implements the default MMA transforms selection for gfx11 targers
 * @tparam MmaOp Mma operation
 * @tparam GfxTargetId Graphics target identifier
 */
template <MmaOpI MmaOp, uint32_t GfxTargetId>
struct MmaTransformsDefaultSelector<MmaOp, GfxTargetId, enable_if_gfx11_target_id_t<GfxTargetId>>
{
    using SelectedTransforms = MmaDefaultTransformsGfx11;
};

/*! @struct MmaTransformsDefaultSelector
 * @brief Implements the default MMA transforms selection for gfx12 targers
 * @tparam MmaOp Mma operation
 * @tparam GfxTargetId Graphics target identifier
 */
template <MmaOpI MmaOp, uint32_t GfxTargetId>
struct MmaTransformsDefaultSelector<MmaOp, GfxTargetId, enable_if_gfx12_target_id_t<GfxTargetId>>
{
    using SelectedTransforms = MmaDefaultTransformsGfx12;
};

} // namespace ck::tile::core::arch::mma
