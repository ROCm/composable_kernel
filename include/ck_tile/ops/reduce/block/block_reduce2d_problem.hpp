// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename XDataType_, typename ComputeDataType_, typename BlockShape_>
struct BlockReduce2dProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
};

} // namespace ck_tile
