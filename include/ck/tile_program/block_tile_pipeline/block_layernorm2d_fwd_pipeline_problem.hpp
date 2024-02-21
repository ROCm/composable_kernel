// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/type.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename XDataType_,
          typename GammaDataType_,
          typename BetaDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename MeanDataType_,
          typename InvStdDataType_,
          index_t kBlockSize_,
          typename BlockLayernorm2dFwdShape_>
struct BlockLayernorm2dFwdPipelineProblem
{
    using XDataType                = remove_cvref_t<XDataType_>;
    using GammaDataType            = remove_cvref_t<GammaDataType_>;
    using BetaDataType             = remove_cvref_t<BetaDataType_>;
    using ComputeDataType          = remove_cvref_t<ComputeDataType_>;
    using YDataType                = remove_cvref_t<YDataType_>;
    using MeanDataType             = remove_cvref_t<MeanDataType_>;
    using InvStdDataType           = remove_cvref_t<InvStdDataType_>;
    using BlockLayernorm2dFwdShape = remove_cvref_t<BlockLayernorm2dFwdShape_>;

    static constexpr index_t kBlockSize = kBlockSize_;
};

} // namespace block
} // namespace tile_program
} // namespace ck
