// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/host/concat.hpp"

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

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', 
            BlockShape::GetName(),
            "op",
            ElementWiseOperation::name,
            "kPad",
            kPad
        );
        // clang-format on
    }
};

} // namespace ck_tile
