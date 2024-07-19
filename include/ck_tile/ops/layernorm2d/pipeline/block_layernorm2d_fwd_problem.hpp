// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename XDataType_,
          typename GammaDataType_,
          typename BetaDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename MeanDataType_,
          typename InvStdDataType_,
          typename BlockShape_>
struct BlockLayernorm2dFwdProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using GammaDataType   = remove_cvref_t<GammaDataType_>;
    using BetaDataType    = remove_cvref_t<BetaDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using MeanDataType    = remove_cvref_t<MeanDataType_>;
    using InvStdDataType  = remove_cvref_t<InvStdDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
};

} // namespace ck_tile
