// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "mfma.hpp"

namespace ck::tile::core::arch::mma {

/*! @struct MmaDefaultTransformsGfx9
 * @brief Implements the default MMA transforms for gfx9 targets
 */
struct MmaDefaultTransformsGfx9
{
    using TransformA = PassThroughTransform;
    using TransformB = PassThroughTransform;
    using TransformC = PassThroughTransform;
    using TransformD = PassThroughTransform;
};

/*! @struct MmaTransformsDefaultSelector
 * @brief Implements the default MMA transforms selection for gfx9 targets
 * @tparam MmaOp Mma operation
 * @tparam GfxTargetId Graphics target id
 */
template <MmaOpI MmaOp, uint32_t GfxTargetId>
struct MmaTransformsDefaultSelector<MmaOp, GfxTargetId, enable_if_gfx9_target_id_t<GfxTargetId>>
{
    using SelectedTransforms = MmaDefaultTransformsGfx9;
};

} // namespace ck::tile::core::arch::mma
