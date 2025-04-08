// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// Y = X * SmoothScale, QY = RowwiseDynamicQuant(Y) = SaturateCast(Y / YScale)
template <typename XDataType_,
          typename ScaleDataType_,
          typename ComputeDataType_,
          typename QXDataType_,
          typename BlockShape_,
          bool kPadN_>
struct PerTensorQuantPipelineProblem
{
    using XDataType           = remove_cvref_t<XDataType_>;
    using ScaleDataType       = remove_cvref_t<ScaleDataType_>;
    using ComputeDataType     = remove_cvref_t<ComputeDataType_>;
    using QXDataType          = remove_cvref_t<QXDataType_>;
    using BlockShape          = remove_cvref_t<BlockShape_>;

    static constexpr bool kPadN    = kPadN_;

    static constexpr bool kNeedCrossWarpSync = BlockShape::WarpPerBlock_N > 1;

};

} // namespace ck_tile
