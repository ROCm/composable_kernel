// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename InDataType_, typename OutDataType_, typename BlockShape_, index_t NDimSpatial_>
struct BlockImageToColumnProblem
{
    using InDataType  = remove_cvref_t<InDataType_>;
    using OutDataType = remove_cvref_t<OutDataType_>;
    using BlockShape  = remove_cvref_t<BlockShape_>;

    static constexpr index_t NDimSpatial = NDimSpatial_;
};

} // namespace ck_tile
