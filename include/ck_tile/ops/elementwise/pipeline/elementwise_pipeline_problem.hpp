// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename XDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename BlockShape_,
          typename ElementWiseOperation_,
          bool kPad_ = true>
struct ElementWisePipelineProblem
{
    using XDataType            = remove_cvref_t<XDataType_>;
    using ComputeDataType      = remove_cvref_t<ComputeDataType_>;
    using YDataType            = remove_cvref_t<YDataType_>;
    using BlockShape           = remove_cvref_t<BlockShape_>;
    using ElementWiseOperation = remove_cvref_t<ElementWiseOperation_>;
    static constexpr bool kPad = kPad_;
};

} // namespace ck_tile
